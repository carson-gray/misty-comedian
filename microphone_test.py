import pyaudio
import wave

joke_id='joke'
duration=10
chunk=1024
channels=1
rate=44100

format = pyaudio.paInt16
wav_fname = "microphone_test.wav"

p = pyaudio.PyAudio()

# Reads microphone stream
stream = p.open(format=format,
                channels=channels,
                rate=rate,
                input=True,
                frames_per_buffer=chunk)

frames = []

for i in range(0, int(rate / chunk * duration)):
    data = stream.read(chunk)
    frames.append(data)

# self.get_decibels(data)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(wav_fname, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(format))
wf.setframerate(rate)
wf.writeframes(b''.join(frames))
wf.close()