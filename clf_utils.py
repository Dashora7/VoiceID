import os
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.linear_model import Perceptron

de = r'C:\Users\nrdas\Downloads\VoiceID'
os.chdir(de)

class VoiceIDSys:
    def __init__(self):
        # A user dictionary with name:centroid
        self.users = {}
        self.encoder = VoiceEncoder()

    def save_sounds(self, duration=10, fs=4100):
        print('recording now!')
        sample = sd.rec(int(duration * fs), samplerate=fs, channels=2)
        sd.wait()
        print('recording done!')
        path = 'output.wav'
        write(path, fs, sample)
        return sample, path

    def generate_voice_profile(self, data_path):
        embeds = []
        os.chdir(data_path)
        for file in os.listdir('.'):
            fpath = Path(os.getcwd() + '\\' + file)
            wav = preprocess_wav(fpath)
            embed = self.encoder.embed_utterance(wav)
            embeds.append(embed)
        centroid = np.array(embeds).mean(axis=0)
        os.chdir(de)
        return centroid

    def add_user(self, name, centroid):
        self.users[name] = centroid

    def id_subject(self, voicepath, th=0.45):
        fpath = Path(voicepath)
        wav = preprocess_wav(fpath)
        embedding = self.encoder.embed_utterance(wav)
        diff_list = []
        for i in self.users.values():
            diff_list.append(i - embedding)
        norms = np.linalg.norm(diff_list, axis=1)
        print(norms)
        m = min(norms)
        if m > th:
            print('Unauthorized')
            print(m)
            return False
        else:
            user = list(self.users.keys())[np.argmin(norms)]
            print('Hello', user, '!')
            return True

if __name__=='__main__':

    sys = VoiceIDSys()
    my_profile = sys.generate_voice_profile('C:\\Users\\nrdas\\Downloads\\VoiceID\\data\\nitish')
    my_profile2 = sys.generate_voice_profile('C:\\Users\\nrdas\\Downloads\\VoiceID\\data\\nitish_on_new_mic')
    sys.add_user('nitish', my_profile)
    sys.add_user('nitishv2', my_profile2)
    voice, path = sys.save_sounds()
    sys.id_subject(path)
    os.remove(path)