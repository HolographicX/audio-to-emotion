from python_speech_features import mfcc
import matplotlib.pyplot as plt
import os
import numpy as np
import librosa
import librosa.display
import scipy.io.wavfile as wav
from matplotlib import cm

list_emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

file = 'dataset/Actor_01/03-01-06-01-01-01-01.wav'

data, sampling_rate = librosa.load( file, sr=44100)
basename = os.path.basename(file)
emotion = list_emotion[basename.split("-")[2]]

plt.title(f'Sound wave of {emotion}')
librosa.display.waveshow(data, sampling_rate)
# plt.show()

D = np.abs(librosa.stft(data))
librosa.display.specshow(D, sr=sampling_rate, x_axis='time', y_axis='linear');
plt.colorbar()
# plt.show()

DB = librosa.amplitude_to_db(D, ref=np.max)
librosa.display.specshow(DB, sr=sampling_rate, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f db')
plt.show()

plt.magnitude_spectrum(data, scale='dB')
# plt.show()

(rate,sig) = wav.read(file)
mfcc_data = mfcc(sig,rate)
fig, ax = plt.subplots()
mfcc_data= np.swapaxes(mfcc_data, 0 ,1)
cax = ax.imshow(mfcc_data, interpolation='nearest', cmap=cm.coolwarm, origin='lower', aspect='auto')

ax.set_title('MFCC')
# plt.xlim([0, 50])
plt.xlabel("Time (ms)")
plt.ylabel("MFCC Coefficients")
plt.show()