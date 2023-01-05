import requests
import json
import os
import sys
import atexit
import pyaudio
import wave
import cv2
import subprocess
import shutil
import pandas as pd
import numpy as np
import sys
import pickle
import time
import datetime
import audioop
from random import randint
from threading import Thread
from websocket import create_connection
from websocket._exceptions import WebSocketConnectionClosedException

class MistyComedian():
    """ Turn the Misty II robot into a comedian """
    def __init__(self, tag_condition='adaptive', subject_number='000'):
        self.SUBJECT_NUMBER = subject_number
        
        # get tag condition from args
        self.TAG_CONDITION = tag_condition.lower()

        # make sure tag condition is valid
        if self.TAG_CONDITION=='none' or self.TAG_CONDITION=='adaptive' or self.TAG_CONDITION=='default':
            pass
        else:
            print(f"ERROR: tag condition {self.TAG_CONDITION} is invalid. Please use 'adaptive', 'default', or 'none'.")
            sys.exit()

        # load basic settings/resources
        self.load_settings()
        self.load_facs()
        self.load_performance()

        atexit.register(self.controlled_shutdown)
        
        # acquired through get_audio_baseline.py prior to performance
        self.set_audio_baseline()
        # flag that video processing uses, that is toggled by audio thresholding
        self.reading_response = True

        # attempt handshakes
        self.connect_to_misty()
        self.reset_face()

        # start video stream
        self.start_av_stream()

        # open stream in OpenCV
        self.VIDEO = cv2.VideoCapture(self.STREAM_URL)
        
        # create a thread to maintain latest frame (needed to prevent buffer lag)
        self.VT = Thread(target=self.frame_manager)
        self.VT.setDaemon(True)
        self.VT.start()
        print("LOG: Running stream through OpenCV")

        # subscribe to websocket to say when audio play has completed
        # this allows for automatic timing
        self.audioplaycomplete_websocket()
        self.closing_ws = False
        self.audioplaycomplete = False
        self.AUDIOPLAYCOMPLETE = Thread(target=self.audioplaycomplete_manager)
        self.AUDIOPLAYCOMPLETE.setDaemon(True)
        self.AUDIOPLAYCOMPLETE.start()

        # openface executable
        self.OPENFACE = os.path.join(os.getcwd(), 'openface_binaries', 'FeatureExtraction.exe')

        if os.path.exists(self.OPENFACE) == False:
            print("Download the openface binaries at https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation#binaries")
            print("Put the binaries here: 'misty-comedian/openface_binaries'")
            sys.exit()

        # column names for audio classification
        self.AUDIO_FEATURES = [
            'length', 'intensity_mean', 'intensity_std', 'intensity_min', 'intensity_max',
            'pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max'
        ]

        # load audio classifier
        with open(os.path.join(os.getcwd(), 'laughter_clf', self.CLASSIFIER), 'rb') as f:
            self.AUDIO_CLF = pickle.load(f)

        # load video classifier (logistic regression with above features)
        with open('face_model.p', 'rb') as f:
            self.VIDEO_CLF = pickle.load(f)


    def controlled_shutdown(self):
        """ Triggers at program exit """
        self.shutdown_websocket()
        self.stop_av_stream()
        self.reset_face()

        return


    def load_settings(self, file='settings.json'):
        """ Load settings json """
        try:
            with open(file, 'r') as f:
                self.settings = json.load(f)
                return
        except FileNotFoundError:
            print(f"ERROR: Could not find '{file}' in '{os.getcwd()}'")
            sys.exit()
    

    def set_audio_baseline(self): 
        try:
            mean = float(self.settings['audio']['mean'])
            std = float(self.settings['audio']['std'])
        except KeyError as e:
            print(f"ERROR: settings file does not have key {e}")
            sys.exit()

        # percentile values for z
        # 99: 2.33, 95: 1.65, 90: 1.28
        z = 2.33
        self.AUDIO_BASELINE = mean + z * std
        return


    def load_facs(self, file='facs.csv'):
        """ Load the facial action coding system reference """
        try:
            with open(file, 'r') as f:
                self.facs_df = pd.read_csv(f)
                self.action_units = ['AU01_c','AU02_c','AU04_c','AU05_c','AU06_c','AU07_c','AU09_c','AU10_c','AU12_c','AU14_c','AU15_c','AU17_c','AU20_c','AU23_c','AU25_c','AU26_c','AU28_c','AU45_c']
                return
        except FileNotFoundError:
            print(f"ERROR: Could not find '{file}' in '{os.getcwd()}'")
            sys.exit()


    def connect_to_misty(self):
        """ Loads Misty's IP address and tries a simple API call """
        try:
            # get IP address from settings file
            self.IP_ADR = str(self.settings['misty']['ipAddress'])
        except KeyError as e:
            print(f"ERROR: settings file does not have key {e}")
            sys.exit()

        # try a simple call
        # TODO: this returns a Timeout error as well as a few more errors if Misty is off
        r = requests.get(url='http://'+self.IP_ADR+'/api/battery')
        
        # if a successful 2xx status code is returned
        if str(r.status_code)[0] == '2':
            print("\nLOG: Connected to Misty")
            return
        else:
            print(f"\nERROR: Test call to Misty failed with status code {r.status_code}, check your IP address in settings.json")
            sys.exit()


    def load_performance(self):
        """ Loads performance file and tests joke files """
        try:
            self.PERFORMANCE = str(self.settings['performance']['setlist'])
            self.JOKES_DIR = str(self.settings['performance']['jokesDir'])
            self.CLASSIFIER = self.settings['performance']['laughterClassifier']

        except KeyError as e:
            print(f"ERROR: settings file does not have key {e}")
            sys.exit()

        # folder to archive the performance
        now = datetime.datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self.PERF_PATH = os.path.join('archived', self.SUBJECT_NUMBER +'_'+ self.TAG_CONDITION + '_' + self.PERFORMANCE[13:-4] + '_' + now)
        try:
            os.mkdir(self.PERF_PATH)
        except FileExistsError:
            shutil.rmtree(self.PERF_PATH)
            os.mkdir(self.PERF_PATH)


    def start_av_stream(self):
        """ Start Misty's AV stream """
        try:
            # get streaming port from settings file
            self.PORT = str(self.settings['misty']['port'])
        except KeyError as e:
            print(f"ERROR: settings file does not have key {e}")
            sys.exit()
        
        # set up parameters
        self.STREAM_URL = 'rtsp://'+self.IP_ADR+':'+self.PORT
        self.STREAM_PARAMS = {"URL": "rtspd:"+self.PORT, "Width": 640, "Height": 480}

        # attempt to start streaming
        r1 = requests.post('http://'+self.IP_ADR+'/api/services/avstreaming/enable')  # this enables the possibility of streaming
        r2 = requests.post('http://'+self.IP_ADR+'/api/avstreaming/start', params=self.STREAM_PARAMS)  # this starts the actual stream

        # if a successful 2xx status code is returned
        if str(r1.status_code)[0] == str(r2.status_code)[0] == '2':
            # You can watch the stream using VLC Media Player
            print(f"LOG: Successfully started streaming at {self.STREAM_URL}")
        else:
            print("ERROR: Unable to start streaming")
            sys.exit()

        return
    

    def stop_av_stream(self):
        """ Triggers on program shutdown to stop the AV stream """
        requests.post('http://'+self.IP_ADR+'/api/avstreaming/stop')  # stop the stream
        requests.post('http://'+self.IP_ADR+'/api/services/avstreaming/disable')  # disable the possibility of streaming


    def frame_manager(self):
        """ Continuously maintains latest video frame """
        while True:
            ret, self.raw = self.VIDEO.read()
            if not ret:
                pass


    def audioplaycomplete_manager(self):
        """ Continually listens for the AudioPlayComplete event from Misty """
        # flag triggered on shutdown
        while self.closing_ws == False:
            try:
                result = self.ws.recv()
            except WebSocketConnectionClosedException:
                # websocket closes on shutdown
                pass

            if result:
                result_dict = json.loads(result)
                if type(result_dict['message']) == dict:
                    # joke has finished playing
                    # print(f"LOG: Finished playing {result_dict['message']['metaData']['name']}")
                    self.audioplaycomplete = True
                
                elif result_dict['message'] == "Registration Status: API event registered.":
                    # websocket has been registered
                    print(f"LOG: Successfully subscribed to AudioPlayComplete event")
                
                elif result_dict['message'] == "Registration Status: Cannot register an event with same name (AudioPlayComplete) as a previously registered event.":
                    # websocket failed to register
                    print(f"ERROR: {result_dict['message']}")
                    print("ERROR: Websocket not properly closed prior to this performance, please restart program")
                
                
    def audioplaycomplete_websocket(self):
        """ Subscribes to the Misty event that is triggered when audio finishes playing """
        self.ws = create_connection("ws://"+self.IP_ADR+"/pubsub")

        # sometimes websocket doesn't shut down properly on premature exit, which requires robot restart
        # by adding a random int to the event name, we can avoid this problem in the short term
        # however, a restart should still be done at some point to avoid it getting bogged down
        # when this issue occurs, it is flagged by the manager function
        event_name = "AudioPlayComplete_"+str(randint(10,99))

        sub_msg = {
            "Operation": "subscribe",
            "Type": "AudioPlayComplete",
            "EventName": event_name,
            "DebounceMS": 33
        }

        sub_msg = json.dumps(sub_msg, separators=(',', ':'))
        self.ws.send(sub_msg)


    def shutdown_websocket(self):
        try:
            # this flag stops audioplaycomplete_manager from spazzing out on shutdown (repeating last message a zillion times)
            # the last joke's message is still repeated once on shutdown, and I believe it is because it is waiting for a message
            # in ws.recv, so the flag isn't checked until it has another pass through the loop. Fixing is low priority
            self.closing_ws = True
            self.ws.close()
        except AttributeError:
            # closed before websocket could be established
            pass


    def wait_for_joke_to_finish(self):
        """ Waits until flag is triggered in audioplaycomplete_manager """
        # wait for audio to finish playing
        self.audioplaycomplete = False
        while self.audioplaycomplete == False:
            pass
        return


    def play_onboard_audio(self, joke_name, volume=80):
        """ Plays preexisting audio off the robot """
        if '.mp3' not in joke_name:
            joke_name += '.mp3'
        
        requests.post(
            url='http://'+self.IP_ADR+'/api/audio/play',
            params={'fileName': joke_name, 'volume': volume}
        )


    def read_the_room(self, joke_id='joke', duration=3, video_fps=4):
        """ Runs the video and audio classification in pseudo parallel """
        # flag that video processing uses, that is toggled by audio thresholding
        # I do not know if this is thread safe
        self.reading_response = True
        
        # start threads that call audio and video methods
        at = Thread(target=self.listen_for_laughter, args=[joke_id, duration])
        at.start()
        vt = Thread(target=self.read_facial_expression, args=[joke_id, duration, video_fps])
        vt.start()
        
        # wait for thread completion, then join them to the main thread
        vt.join()
        at.join()

        # return joke id and audience response
        response = {'joke': joke_id, 'video': self.temp_video_label, 'audio': self.temp_audio_label}
        return response


    def read_facial_expression(self, joke_id='joke', duration=3, fps=10):
        """ Reads participant's facial affect, returning -1/0/1 """
        # access videostream
        self.record_video(joke_id=joke_id, duration=duration, fps=fps)
        
        # run openface on the video frames
        self.extract_action_units()

        # find the label that best matches the openface output
        self.temp_video_label = self.classify_action_units()
        return


    def record_video(self, joke_id='joke', duration=3, fps=10):
        """ Process audience response for 'sec' seconds
        and 'fps' frames per second (max 30)"""
        # Misty's max fps is 30
        if fps > 30:
            fps = 30
        
        num_saved = 0
        frame_count = 0

        # for number of frames the 30fps stream will have over the time interval
        # for i in range (int(duration*30)):

        # Keeps going until audio processing toggles this value
        # I do not know if this is thread safe
        while self.reading_response:
            # only process at desired rate (ex: 3 fps means process every 30/3=10 frames)
            if frame_count % int(30/fps) == 0:
                # # catch a single frame
                # ret, raw = self.VIDEO.retrieve()
                raw = self.raw

                # rotate frame to be upright
                last = cv2.flip(cv2.rotate(raw, cv2.ROTATE_90_CLOCKWISE), flipCode=1)

                # save to the frames folder
                fpath = os.path.join('frames', joke_id+'_'+self.add_zero(num_saved)+'.jpg')
                cv2.imwrite(fpath, last)
                num_saved += 1
            
            frame_count += 1
            
            # wait for 1 frame of Misty's stream
            time.sleep(1/30)

        return

    
    def extract_action_units(self):
        """ Runs openface on images in 'frames' directory, 
        generating output csv 'frames/output.csv' """
        # runs openface executable, which generates 'processed' directory in cwd with 'frames.csv'
        subprocess.run([self.OPENFACE,'-fdir', 'frames'], capture_output=True)

        # set both paths
        in_path = os.path.join('processed', 'frames.csv')
        out_path = os.path.join('frames', 'openface.csv')

        # move resulting file
        shutil.move(in_path, out_path)

        # delete processed directory
        shutil.rmtree('processed')
        
        return


    def classify_action_units(self):
        """ Classifies face by finding the reference action unit combo with
        the min distance from the mean of the respondent's action unit combo """
        # dataframe of person's face
        df = pd.read_csv(os.path.join('frames', 'openface.csv'))
        
        # strip whitespace from column names
        df.columns = df.columns.str.strip()

        # NEW WAY
        # column names for video classification
        preprocessed_features = ['AU06_r', 'AU12_r', 'AU10_r', 'pose_Tx', 'AU14_r']

        # column names for video classification
        postprocessed_features = ['AU06_r_range', 'AU12_r_range', 'AU10_r_range', 'pose_Tx_range', 'AU14_r_range']

        feature_values = []
        
        # iterate over keys within timeseries data file
        for feature in preprocessed_features:
            # get the data for that key (pandas Series object)
            timeseries = df[feature]
            feature_values.append(timeseries.max() - timeseries.min())

        classification_df = pd.DataFrame(data=[feature_values], columns=postprocessed_features)
        
        # classify the dataframe
        label = int(self.VIDEO_CLF.predict(classification_df))
        
        # OLD WAY
        # # find the difference between mean response and each labelled emotion combination
        # # since it is binary 0/1, the mean is the proportion of frames the action unit is 'on'
        # # facs_df is a dataframe where each column is a specific action unit and each row is a labelled emotion combination of action units
        # # diff_df has action unit columns as well, and each row is the difference between that column's action unit for the current reaction and the labelled emotion
        # diff_df = np.abs(self.facs_df[self.action_units] - df[self.action_units].mean())

        # # find the norm of each of these difference vectors
        # # this treats each row (labelled emotion combination) of the dataframe as a vector
        # norm_df = diff_df.apply(np.linalg.norm, axis=1)

        # # classify face as the label with shortest norm
        # label = self.facs_df.loc[norm_df.idxmin()]['label']

        return label


    def listen_for_laughter(self, joke_id='joke', duration=3):
        """ Reads participant laughter, returning -1/0/1 """
        # get audio from system microphone
        self.record_audio(joke_id=joke_id, duration=duration)

        # convert wav to dataframe of audio data
        df = self.extract_audio_data(joke_id=joke_id)
        
        # classify the dataframe
        self.temp_audio_label = self.AUDIO_CLF.predict(df[self.AUDIO_FEATURES])[0]
        return


    def extract_audio_data(self, joke_id='joke'):
        """ extract audio information from joke_id.wav using praat """
        praat = os.path.join("laughter_clf", "Praat.exe")
        script = os.path.join("laughter_clf", "extract_basic.praat")
        out_txt = os.path.join("laughter_clf", joke_id+".txt")

        # run command line that takes in joke_id.wav and outputs praat-processed joke_id.txt
        program = praat+' --run '+script+' "'+joke_id+'.wav" > "'+out_txt+'"'
        os.system(program)

        # read the txt file as a list
        with open(out_txt, encoding='utf-16-le') as file:
            lines = file.readlines()
        
        # remove extra space, convert to float
        lines = [line.rstrip() for line in lines]
        
        row = []
        for line in lines:
            if line == '--undefined--':
                row.append(0.0)
            else:
                row.append(float(line))

        # create a dataframe
        df = pd.DataFrame([row], columns=self.AUDIO_FEATURES)

        # return the classification-ready dataframe
        return df


    def record_audio(self, joke_id='joke', duration=3, chunk=1024, channels=1, rate=44100, min_pause=1.5, max_pause=3, volume_interval=5, volume_strikes=3):
        """ Records audio from system microphone, NOT the robot, until sound dies down """
        format = pyaudio.paInt16
        wav_fname = joke_id+".wav"

        p = pyaudio.PyAudio()

        # Reads microphone stream
        stream = p.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

        frames = []
        frame_count = 0
        strike_count = 0
        counts_per_sec = int(rate/chunk)
        rms_arr = np.array([])

        # # This for loop is if you wanna do it by 'duration' seconds
        # for i in range(0, int(rate / chunk * duration)):

        # read the audio stream until the sound dies down, with a guaranteed minimum pause time
        while strike_count < volume_strikes or frame_count < (counts_per_sec*min_pause):
            # cap pause time at a maximum value
            # this is useful when environmental noise increases, making baseline ineffective
            if frame_count > (counts_per_sec*max_pause):
                print("Triggered maximum pause length")
                break

            # read in audio data
            data = stream.read(chunk)
            
            # grab audio frame to be put in wav file later
            frames.append(data)
            
            # get the power of the audio frame (proxy for volume)
            rms = audioop.rms(data, 2)
            rms_arr = np.append(rms_arr, rms)
            frame_count += 1

            # check volume over "1 second/volume_interval" seconds
            if frame_count % int(counts_per_sec/volume_interval) == 0:
                mean = np.mean(rms_arr)
                rms_arr = np.array([])

                # compare against baseline
                # if below baseline, get a strike
                if mean < self.AUDIO_BASELINE:
                    strike_count += 1

        # set flag for video processor
        # I do not know if this method is thread safe
        self.reading_response = False

        # stop taking in audio input
        stream.stop_stream()
        stream.close()
        p.terminate()

        # write to a wave file for classification
        wf = wave.open(os.path.join('laughter_clf', wav_fname), 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()


    def classify_response(self, response):
        """ Simple classification for overall response """
        video = response['video']
        audio = response['audio']

        # only using video for classification
        return video


    def archive_response(self, joke_id="joke", response=""):
        """ Moves the video frames and openface output into the archived folder """
        # move it into a folder that says the joke id and when it happened
        path = os.path.join(self.PERF_PATH, joke_id)
        try:
            os.mkdir(path)
        except FileExistsError:
            shutil.rmtree(path)
            os.mkdir(path)

        for fname in os.listdir('frames'):
            src = os.path.join('frames', fname)
            shutil.move(src, os.path.join(path, fname))
        
        for fname in os.listdir('laughter_clf'):
            if fname.endswith('.txt') or fname.endswith('.mp3') or fname.endswith('.wav'):
                src = os.path.join('laughter_clf', fname)
                shutil.move(src, os.path.join(path, fname))
        
        with open(os.path.join(path, 'response.txt'), 'w') as f:
            f.write(str(response))


    def add_zero(self, num):
        """ adds a zero if single digit, converts to str """
        if num < 10:
            return '0' + str(num)
        else:
            return str(num)


    def react_face(self, classification=0):
        """ Plays the appropriate physical response based on classification result """
        if classification == 1:
            # look upwards with her head in pride
            requests.post(
                url='http://'+self.IP_ADR+'/api/head',
                params={
                    "Pitch": -10,
                    "Roll": 0,
                    "Yaw": 0,
                    "Velocity": 100
                }
            )

            # set facial expression to positive
            requests.post(
                url='http://'+self.IP_ADR+'/api/images/display',
                params={'FileName': "e_Admiration.jpg", 'Alpha': 1}
            )
        
        elif classification == 0:
            # set facial expression to neutral
            requests.post(
                url='http://'+self.IP_ADR+'/api/images/display',
                params={'FileName': "e_ApprehensionConcerned.jpg", 'Alpha': 1}
            )
        
        elif classification == -1:
            # moves her head to the side in frustration
            requests.post(
                url='http://'+self.IP_ADR+'/api/head',
                params={
                    "Pitch": 0,
                    "Roll": 12,
                    "Yaw": 0,
                    "Velocity": 100
                }
            )

            # set facial expression to negative
            requests.post(
                url='http://'+self.IP_ADR+'/api/images/display',
                params={'FileName': "e_Disgust.jpg", 'Alpha': 1}
            )

        # set facial expression to one holding back a laugh
        # requests.post(
        #     url='http://'+self.IP_ADR+'/api/images/display',
        #     params={'FileName': "e_Joy.jpg", 'Alpha': 1}
        # )   
        
        return


    def reset_face(self):
        # reset head position
        requests.post(
            url='http://'+self.IP_ADR+'/api/head',
            params={
                "Pitch": 0,
                "Roll": 0,
                "Yaw": 0,
                "Velocity": 100
            }
        )

        # reset facial expression
        requests.post(
            url='http://'+self.IP_ADR+'/api/images/display',
            params={'FileName': "e_DefaultContent.jpg", 'Alpha': 1}
        )


    def play_performance(self, time_between_jokes=1, volume=80):
        try:
            with open(self.PERFORMANCE, 'r') as setlist_file:
                for setlist_line in setlist_file:
                    with open(os.path.join(self.JOKES_DIR, setlist_line[0:-1]), 'r') as joke_file:
                        # joke_ssml:str, positive_tag:str, neutral_tag:str, negative_tag:str, default_classification:int, joke_has_tag:bool
                        df = pd.read_csv(joke_file, sep=";")

                        # joke name is csv name
                        joke_name = setlist_line[0:-5]

                        # onboard mp3's with jokename_jokepart.mp3 as filenames
                        self.play_onboard_audio(joke_name=joke_name+"_joke", volume=volume)
                        self.wait_for_joke_to_finish()

                        # TODO: Add midjoke reading functionality, likely by incorporating into wait_for_joke_to_finish()
                        # You can have some sort of flag for separating mid and post in real time
                        if df['joke_has_tag'].values[0] == True:

                            response = self.read_the_room(joke_id=joke_name, duration=3)
                            classification = self.classify_response(response)
                            response['classification'] = classification
                            print(response)
                            
                            # adaptive mode reacts based on classification
                            if self.TAG_CONDITION == 'adaptive':
                                self.react_face(classification=classification)

                                if classification == 1:
                                    self.play_onboard_audio(joke_name=joke_name+"_positive", volume=volume)
                                elif classification == 0:
                                    self.play_onboard_audio(joke_name=joke_name+"_neutral", volume=volume)
                                elif classification == -1:
                                    self.play_onboard_audio(joke_name=joke_name+"_negative", volume=volume)

                                self.wait_for_joke_to_finish()
                                self.reset_face()
                            
                            # default mode reacts with a prescripted response
                            elif self.TAG_CONDITION == 'default':
                                default = df['default_classification'].values[0]
                                self.react_face(classification=default)

                                if default == 1:
                                    self.play_onboard_audio(joke_name=joke_name+"_positive", volume=volume)
                                elif default == 0:
                                    self.play_onboard_audio(joke_name=joke_name+"_neutral", volume=volume)
                                elif default == -1:
                                    self.play_onboard_audio(joke_name=joke_name+"_negative", volume=volume)

                                self.wait_for_joke_to_finish()
                                self.reset_face()

                            # none mode does not use tags at all
                            elif self.TAG_CONDITION == 'none':
                                pass

                            # TODO: potentially add a short read following the tag

                            response['performance_tag_condition'] = self.TAG_CONDITION
                            response['joke_has_tag'] = df['joke_has_tag'].values[0]
                            self.archive_response(joke_id=joke_name, response=response)
                        
                        if df['joke_has_tag'].values[0] == False and self.TAG_CONDITION == 'none':
                            time.sleep(time_between_jokes)

                        if self.TAG_CONDITION != 'none':
                            time.sleep(time_between_jokes)
                return
        except FileNotFoundError as e:
            print(e)
            print("Be sure there is an empty last line in performances txt document")
            sys.exit()


if __name__ == "__main__":
    #TODO: after booting up misty, gotta do a dry run otherwise there will be a big pause after first joke. idk why
    input("\nPress enter to indicate that you have taken the audio baseline for the room! ")

    # set tag condition
    while True:
        tag_condition = input("Choose tag condition:\nadaptive [a], default [d], or none [n]: ").lower()
        
        # set tag_condition based on shorthand
        if tag_condition == 'n':
            tag_condition = 'none'
        elif tag_condition == 'd':
            tag_condition == 'default'
        elif tag_condition == 'a':
            tag_condition = 'adaptive'

        # make sure tag condition is valid (if they typed the full value or something different)
        if tag_condition=='none' or tag_condition=='adaptive' or tag_condition=='default':
            # if it is valid, break the loop
            break
        else:
            # go back to top
            print("Please enter either a, d, or n\n")
    
    subject_number = input("What is the subject number? ")
    
    # get Misty booted up
    misty = MistyComedian(tag_condition=tag_condition, subject_number=subject_number)

    # wait for user input to start
    input("\nPress enter to start the performance! ")

    # time for research assistant to leave the room
    time.sleep(5)

    # start the show!
    misty.play_performance(time_between_jokes=1, volume=70)
