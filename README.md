# Decoding Emotions from Audio Recordings
## Background
Fourier transforms have the simple (yet mathematically complex) purpose of breaking down waveforms into an alternate form of varying sine and cosine functions. Essentially, it depicts that any waveform (a function of time) can be re-written as a sum of sinusoidals. A common use of Fourier Transforms is in Signal Processing, processing any waveform (which could be light waves, your speech, or even stocks) to extract relevant information. For example, you could “filter out” unwanted parts of a signal by breaking it down, such as removing distracting background sounds or digital noise in a photograph. Despite this concept having a presumably narrow focus, the Fourier Transform abounds in applications in seemingly unrelated areas of Math and Physics, like the Uncertainty Principle3 or even the Riemann Zeta Function. For this project, I wanted to take advantage of this ubiquity of Fourier Transforms. Using Fourier Transforms and a feature extraction method using MFCCs, we can train an SKlearn model in Python on the emotion-labelled features. 
Please read the [full report](https://github.com/HolographicX/audio-to-emotion/blob/main/fourier-transforms-emotion-extended-essay.pdf) that intends to explain the mathematical foundation for such a project, and was used for my IB Extended Essay.

# Results 
![image](https://user-images.githubusercontent.com/73580740/209468794-7dc51a5b-d2d3-43c0-a323-73b3df6d3991.png)

74.4% accuracy was acheived in this particular test, with 504 training samples and 168 predictions.

# Credits
- [Tushar Gupta - Speech Emotion Detection](https://medium.com/@tushar.gupta_47854/speech-emotion-detection-74337966cf2)
- [Abdeladim Fadheli - How to Make a Speech Emotion Recognizer Using Python And Scikit-learn](https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn)
