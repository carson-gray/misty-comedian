import pyaudio
import audioop
import numpy as np
import json
import os
import time

while True:
    try:
        duration=int(input("How many seconds should be used for the baseline? "))
        break
    except ValueError:
        print("Please enter a number\n")

chunk=1024
channels=1
rate=44100
format = pyaudio.paInt16

p = pyaudio.PyAudio()

audio_power = []

# Reads microphone stream
stream = p.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk)

print(f"\nRecording baseline audio for {duration} seconds.")
print("Please be quiet for the recording process.")

input("\nPress enter to begin recording. ")
time.sleep(1)
print("3")
time.sleep(1)
print("2")
time.sleep(1)
print("1")
time.sleep(1)
print("\nRecording audio baseline.")

# saves the rms, which is a way of measuring audio power
for i in range(0, int(rate / chunk * duration)):
    data = stream.read(chunk)
    rms = audioop.rms(data, 2)
    audio_power.append(rms)

audio_power = np.array(audio_power)

print("Recording complete! Writing results to settings.json")

# stop the audio stream
stream.stop_stream()
stream.close()
p.terminate()

# update settings to include mean and std of audio power
filename = 'settings.json'
with open(filename, 'r') as f:
    data = json.load(f)
    data['audio']['mean'] = np.mean(audio_power)
    data['audio']['std'] = np.std(audio_power)

# replace settings file with updated file
os.remove(filename)
with open(filename, 'w') as f:
    json.dump(data, f, indent=4)
