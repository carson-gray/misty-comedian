from boto3 import Session
from botocore.exceptions import ProfileNotFound, ClientError, BotoCoreError
from contextlib import closing
from tempfile import gettempdir
import base64
import requests
import os
import json
import sys
import pandas as pd
import time

class MistyJokeSynthesis():
    def __init__(self):
        # load basic settings/resources
        self.load_settings()
        self.load_performance()
        
        # attempt handshakes
        self.connect_to_misty()
        self.connect_to_aws()


    def load_settings(self, file='settings.json'):
        """ Load settings json """
        try:
            with open(file, 'r') as f:
                self.settings = json.load(f)
                return
        except FileNotFoundError:
            print(f"ERROR: Could not find '{file}' in '{os.getcwd()}'")
            sys.exit()


    def load_performance(self):
        """ Loads performance file and points to joke files """
        try:
            # get IP address from settings file
            self.PERFORMANCE = str(self.settings['performance']['setlist'])
            self.JOKES_DIR = str(self.settings['performance']['jokesDir'])
        except KeyError as e:
            print(f"ERROR: settings file does not have key {e}")
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
        try:
            requests.get(url='http://'+self.IP_ADR+'/api/battery')
        except TimeoutError:
            print("ERROR: Could not reach Misty. Please check IP address in settings.json")
            sys.exit()


    def connect_to_aws(self):
        """ Logs into AWS and tries a Polly call """
        try:
            # attempt to load from settings
            profile_name = self.settings['aws']['user']
            region_name = self.settings['aws']['region']
            self.VOICE = self.settings['aws']['pollyVoice']

        except KeyError as e:
            print(f"ERROR: settings file does not have key {e}")
            sys.exit()

        try:
            # attempt to create an AWS session
            self.AWS = Session(profile_name=profile_name, region_name=region_name)
            print("LOG: Created AWS session")

        except ProfileNotFound as e:
            print(f"ERROR [AWS credentials]: {e}")
            print("\nBe sure you have an AWS account with a valid public/private key pair.")
            print("https://docs.aws.amazon.com/powershell/latest/userguide/pstools-appendix-sign-up.html")
            print("\nOnce you have your keys, configure your AWS credentials in your IDE.")
            print("Scroll to 'SDKs & Toolkits' and choose your IDE: https://docs.aws.amazon.com/index.html")
            sys.exit()

        try:
            # create an Amazon Polly client
            self.T2S = self.AWS.client("polly")
        
            # Test connection
            self.T2S.synthesize_speech(Text="test", OutputFormat="mp3", VoiceId=self.VOICE)
            print("LOG: Connected to AWS Polly")

        except (BotoCoreError, ClientError) as e:
            print(e)
            sys.exit()
        
        return  


    def aws_text_to_speech(self, joke_text='<speak>joke text</speak>', joke_name='polly'):
        """ Does text to speech, saying whatever is in
        the 'joke_text' arg. If it breaks, it returns False """
        try:
            # Request speech synthesis
            response = self.T2S.synthesize_speech(Text=joke_text, TextType='ssml', OutputFormat="mp3", VoiceId=self.VOICE)
        except (BotoCoreError, ClientError) as error:
            # The service returned an error, exit gracefully
            print(error)
            return False

        # Access the audio stream from the response
        if "AudioStream" in response:
            with closing(response["AudioStream"]) as stream:
                output = os.path.join(gettempdir(), "speech.mp3")
                try:
                    # Open a file for writing the output as a binary stream
                    with open(output, "wb") as file:
                        file.write(stream.read())
                except IOError as error:
                    # Could not write to file, exit gracefully
                    print(error)
                    return False
        else:
            # The response didn't contain audio data, exit gracefully
            print("Could not stream audio")
            return False

        with open(output, 'rb') as f:
            b = base64.b64encode(f.read())

        requests.post('http://'+self.IP_ADR+'/api/audio',
            params={
                'FileName': joke_name+'.mp3',
                'Data': b,
                'ImmediatelyApply': 'false',
                'OverwriteExisting': 'true'
            }
        )

        # give time for Misty to receive/save file
        time.sleep(5)

        return

    
    def generate_performance(self):
        try:
            with open(self.PERFORMANCE, 'r') as setlist_file:
                print("LOG: Running files")
                for setlist_line in setlist_file:
                    print("\t"+setlist_line[0:-1])

                    with open(os.path.join(self.JOKES_DIR, setlist_line[0:-1]), 'r') as joke_file:
                        # joke name is csv name
                        joke_name = setlist_line[0:-5]

                        # joke_ssml:str, positive_tag:str, neutral_tag:str, negative_tag:str, default_classification:int, joke_has_tag:bool
                        df = pd.read_csv(joke_file, sep=";")

                        # generate mp3's on misty with jokename_jokepart.mp3 as filenames
                        try:
                            self.aws_text_to_speech(joke_text=df['joke_ssml'].values[0], joke_name=joke_name+"_joke")
                        except ConnectionResetError:
                            print("\t\tERROR: Joke is too long to upload. Please shorten.")
                        
                        try:
                            self.aws_text_to_speech(joke_text=df['positive_tag'].values[0], joke_name=joke_name+"_positive")
                        except ConnectionResetError:
                            print("\t\tERROR: Positive tag is too long to upload. Please shorten.")
                        
                        try:
                            self.aws_text_to_speech(joke_text=df['neutral_tag'].values[0], joke_name=joke_name+"_neutral")
                        except ConnectionResetError:
                            print("\t\tERROR: Neutral tag is too long to upload. Please shorten.")
                            
                        try:
                            self.aws_text_to_speech(joke_text=df['negative_tag'].values[0], joke_name=joke_name+"_negative")
                        except ConnectionResetError:
                            print("\t\tERROR: Negative tag is too long to upload. Please shorten.")

                return
        except FileNotFoundError as e:
            print(e)
            print("Be sure there is an empty last line in performances txt document")
            sys.exit()


if __name__ == "__main__":
    # get Misty booted up
    misty_tts = MistyJokeSynthesis()

    # generate and load performance audio
    misty_tts.generate_performance()