form Get files
	text First_text hello
endform
Read from file: first_text$
sound = selected("Sound")

# Length of the sound in seconds
selectObject: sound
sound_duration = Get total duration
appendInfoLine: sound_duration

# Intensity
selectObject: sound
To Intensity: 75, 0
int_mean = Get mean... 0 0 dB
appendInfoLine: int_mean
int_std = Get standard deviation... 0 0
appendInfoLine: int_std
int_min = Get minimum... 0 0 Sinc70
appendInfoLine: int_min
int_max = Get maximum... 0 0 Sinc70
appendInfoLine: int_max

# Pitch
selectObject: sound
To Pitch: 0, 75, 300
pitch_mean = Get mean... 0 0 Hertz
appendInfoLine: pitch_mean
pitch_std = Get standard deviation... 0 0 Hertz
appendInfoLine: pitch_std
pitch_min = Get minimum: 0, 0, "Hertz", "Parabolic"
appendInfoLine: pitch_min
pitch_max = Get maximum: 0, 0, "Hertz", "Parabolic"
appendInfoLine: pitch_max
