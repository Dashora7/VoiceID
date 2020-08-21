# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 17:44:07 2020

@author: nrdas
"""
import sounddevice as sd
from pyAudioAnalysis import audioSegmentation as ag
from pyAudioAnalysis import audioBasicIO as aIO
from scipy.io.wavfile import write

duration = 10
fs = 44100

print('recording now!')
sample = sd.rec(int(duration*fs), samplerate=fs, channels=2)
sd.wait()
print('recording done!')
print(type(sample))
write('output.wav', fs, sample)

# This gets event segments correctly but it is very sensitive
[Fs, x] = aIO.read_audio_file("output.wav")
segments = ag.silence_removal(x, Fs, 0.020, 0.020, smooth_window=1.0, weight=0.3)
print(segments)
