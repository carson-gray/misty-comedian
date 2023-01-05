from boto3 import Session
from botocore.exceptions import ProfileNotFound, ClientError, BotoCoreError
from contextlib import closing
from tempfile import gettempdir
import os
import json
import sys
from playsound import playsound  # pip install playsound==1.2.2
from mutagen.mp3 import MP3

class AWSJokeTester():
    """ Try out joke text in real time without connecting to Misty """
    def __init__(self):
        self.load_settings()
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


    def test_joke(self, joke_text='<speak>joke text</speak>'):
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
        
            # # print the length of the sound file
            # audio = MP3(output)
            # print(f"MP3 Length: {audio.info.length}")

            # automatically play the audio
            playsound(output)
            os.remove(output)
        
        else:
            # The response didn't contain audio data, exit gracefully
            print("Could not stream audio")
            return False

        return


if __name__ == "__main__":
    aws = AWSJokeTester()

    while True:
        user_entry =  input("\nEnter desired SSML text or 'q' to quit: ")

        if user_entry == 'q':
            break

        if "<speak>" not in user_entry:
            user_entry = "<speak>" + user_entry + "</speak>"

        aws.test_joke(joke_text=user_entry)