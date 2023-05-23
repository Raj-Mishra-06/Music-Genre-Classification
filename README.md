# Web-Based-Music-Genre-Classification
A web based music genre classification software that enables user to predict genre of their audio file along with generated result that consists bar graph to show confidence of each genre in that audio file with mel-spectrogram.

music_classification.py file is used to train the model, i have selected 3 clasifcation models K-nearest neighbour, support vector machine, & Mel frequency cepstral coefficient(MFCC) to tarin the clasiifier. from these 3 Mfcc was more accurate with an acccuracy of 70%.

features_30_sec.csv is the extracted features files, these features were extracted through the music_classification.py on GTZAN dataset.

GTZAN dataset consists of 1000 .wav format file for 10 differetnt genres in total, 100 files for each genre of 30 seconds each.
praogramming language was python with Django framework, for frontend i used Html, Css & javascript.

Project Report for github.pdf contains complete documentation of the project.
