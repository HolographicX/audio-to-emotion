from python_speech_features import mfcc
import matplotlib.pyplot as plt
import os
import numpy as np
import librosa
import librosa.display
from scipy.interpolate import make_interp_spline
import scipy.io.wavfile as wav
from matplotlib import cm
import pickle # to save model after training
from sklearn.model_selection import train_test_split # for splitting training and testing
from sklearn.neural_network import MLPClassifier # multi-layer perceptron model
from sklearn.metrics import accuracy_score # to measure how good we are
from sklearn import metrics
import glob
import soundfile


# all emotions on RAVDESS dataset
int2emotion = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

AVAILABLE_EMOTIONS = {
    "angry",
    "fearful",
    "neutral",
    "happy"
}

def extract_feature(file_name):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        X = librosa.to_mono(X)
        sample_rate = sound_file.samplerate
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = (np.hstack(mfccs))
    return result

def load_data(test_size=0.2):
    X, y = [], []
    for file in glob.glob("/home/soham/development/python/audio_to_emotion/dataset/Actor_*/*.wav"):
        # get the base name of the audio file
        basename = os.path.basename(file)
        # get the emotion label
        emotion = int2emotion[basename.split("-")[2]]
        if emotion not in AVAILABLE_EMOTIONS:
            continue

        # extract speech features
        features = extract_feature(file)
        
        # add to data
        X.append(features)
        y.append(emotion)
    # split the data to training and testing and return it
    return train_test_split(np.array(X), y, test_size=test_size, random_state=7)
    # return y

# X_train= load_data(test_size=0.25)
# print(X_train)
X_train, X_test, y_train, y_test = load_data(test_size=0.25)
# print some details
# number of samples in training data
print("[+] Number of training samples:", X_train.shape[0], len(y_train))
# number of samples in testing data
print("[+] Number of testing samples:", X_test.shape[0], len(y_test))

# best model, determined by a grid search
model_params = {
    'alpha': 0.01,
    'batch_size': 256,
    'epsilon': 1e-08, 
    'hidden_layer_sizes': (300,), 
    'learning_rate': 'adaptive', 
    'max_iter': 500, 
}

# initialize Multi Layer Perceptron classifier
model = MLPClassifier(**model_params)


# train the model
print("[*] Training the model...")
model.fit(X_train, np.array(y_train))

# predict 25% of data to measure how good we are
y_pred = model.predict(X_test)

# calculate the accuracy
accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))
confmatrix = metrics.confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confmatrix, display_labels = ["neutral", "happy", "angry", "fearful"])
disp.plot(cmap='Purples')
plt.show()
# now we save the model
# make result directory if doesn't exist yet
if not os.path.isdir("result"):
    os.mkdir("result")

pickle.dump(model, open("result/mlp_classifier.model", "wb"))


#CREDITS (inspiration) - https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn
