import os
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from sklearn.linear_model import Perceptron
os.chdir('C:\\Users\\nrdas\\Downloads\\voiceID\\data\\nitish')
embeds = []
encoder = VoiceEncoder()

for file in os.listdir('.'):
    fpath = Path(os.getcwd() + '\\' + file)
    wav = preprocess_wav(fpath)
    embed = encoder.embed_utterance(wav)
    embeds.append(embed)

embeds2 = []
os.chdir('C:\\Users\\nrdas\\Downloads\\voiceID\\data\\unauthorized')
for file in os.listdir('.'):
    fpath = Path(os.getcwd() + '\\' + file)
    wav = preprocess_wav(fpath)
    embed = encoder.embed_utterance(wav)
    embeds2.append(embed)

centroid = np.array(embeds).mean(axis=0)
diff_dists = embeds2 - centroid
sim_dists = embeds - centroid
sim_dists_norm = np.linalg.norm(sim_dists, axis=1)
diff_dists_norm = np.linalg.norm(diff_dists, axis=1)
print(sim_dists_norm)
print(diff_dists_norm)