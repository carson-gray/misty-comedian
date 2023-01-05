## misty-comedian
Tools for deploying the Misty II robot as a comedian. Please email Carson (carsongray01@gmail.com) if you have any questions!

Also, be sure to check the Misty Comedian System Google docs folder in the SHARE team drive: Commonly Useful Resources/Software and Stats/Robot Specific Resources/Misty

https://docs.google.com/document/d/1A0vF7gSCBwVBm1PYkXyvnx7wSTOyWQbQdZRka3bBqRw/edit

This project works on Windows, but needs modification to work on Mac/Linux

### Python
Used Python 3.9 for pilot and followup studies. I would recommend using a virtual environment for managing libraries between projects.

Run:
`pip install -r reqs.txt`

You may need to use `pip3`

## Connecting to Misty
Set Misty's IP address in 'settings.json'. You can get Misty's IP address using the Misty app, which allows you to connect to Misty via Bluetooth on your phone. Connect Misty to the same wifi as your computer, and then copy the IP address from the app home page into settings.

### Writing jokes
To write jokes, copy jokes/joke_template.csv, change the name to the name of your joke, and fill in the ssml values for the joke, separated by semicolons. 

The default classification item is the label it gives in the hardcoded-tag condition, which can be -1, 0, or 1. 

The joke_has_tag column lets you set whether the joke has tags, either as a True or False (where False only applied to intro/outro in the previous deployment). 

You can test the joke audio without connecting to Misty using aws_joke_tester.py

### Creating performances
To create a new performance, create a txt document in the 'performances' directory. Line by line, write the file names of the jokes you want told, in order. Be sure to leave a space at the end of the file. Point to your desired performance file in the 'settings.json' file's "setlist" key.

### Loading performances onto Misty
To load your performance onto Misty (the one pointed to in settings.json), run misty_joke_synthesis.py

### Setting your audio baseline
To set the baseline for audio volume, run 'get_audio_baseline.py'. This file will automatically fill the results into settings. This should be run prior to your performance.

### Running your performance
Be sure to set you audio baseline first with get_audio_baseline.py

To run your performance, run 'misty_comedian.py'

I HIGHLY recommend running dry_run.txt as the performance after booting up Misty to make sure everything works smoothly. Once Misty has run the dry_run.txt performance (set it in settings.json) change it back to your desired performance and it should work great. 

There is a bug where the first joke after Misty boots up has a very long lag during the classification process. I don't know why, but this workaround fixes it.

### Seeing your results
Performances are saved to the 'archived' folder, where each joke has its video frames and audio, as well as openface output, praat output, and classification decision.

### Changing your audio classifier
Add a pickled audio classifier into the 'laughter_clf' directory and point to it in settings.

### Changing your video classifier
It is currently hardwired in classify_action_units() and init() in misty_comedian.py 

If I have time I will pull it to settings like audio

## Downloading OpenFace
Download the openface binaries at https://github.com/TadasBaltrusaitis/OpenFace/wiki/Windows-Installation#binaries

Extract the folder from the zip folder (in my case, 'OpenFace_2.2.0_win_x64.zip' extracts to 'OpenFace_2.2.0_win_x64'). 

Open Windows powershell as administrator (search for it, right click, and run as admin)

`Set-ExecutionPolicy Unrestricted`

In your downloads folder, right click download_models.ps1 and run in powershell

This will download the models, and will take a while

Once you do that, go back to your admin powershell and set it back to

`Set-ExecutionPolicy Restricted`

Once you have downloaded the models, select all of the folder contents ('amd64', 'AU_predictors', etc), and paste the full contents of the folder into the openface_binaries directory: 'misty-comedian/openface_binaries'

## Downloading Praat
Download Praat at https://www.fon.hum.uva.nl/praat/ and put Praat.exe into the laughter_clf directory.

## Configuring AWS Polly
Be sure you have an AWS account with a valid public/private key pair.

https://docs.aws.amazon.com/powershell/latest/userguide/pstools-appendix-sign-up.html

Once you have your keys, configure your AWS credentials in your IDE. Scroll to 'SDKs & Toolkits' and choose your IDE: https://docs.aws.amazon.com/index.html

To validate that this works, run aws_joke_tester.py. You should be able to type text into the console and get the text's audio over your system speakers.
