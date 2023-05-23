import librosa
import numpy as np
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Specify the path to the GTZAN dataset
dataset_path = 'C:/Users/hp/Desktop/New folder (3)/genres/'

# Check if the specified path exists
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"The specified directory '{dataset_path}' does not exist.")

# Load the GTZAN dataset
csv_file = 'features_30_sec.csv'
csv_path = os.path.join(dataset_path, csv_file)
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"The specified CSV file '{csv_path}' does not exist.")

df = pd.read_csv(csv_path)
X = df.drop(['filename', 'label', 'length'], axis=1).values
y = df['label'].values

audio_files = []
genres = []

for genre in os.listdir(dataset_path):
    genre_path = os.path.join(dataset_path, genre)
    if not os.path.isdir(genre_path):
        continue
    for audio_file in os.listdir(genre_path):
        if not audio_file.endswith(".wav"):
            continue
        audio_file_path = os.path.join(genre_path, audio_file)
        audio_files.append(audio_file_path)
        genres.append(genre)

df = pd.DataFrame({'filename': audio_files, 'label': genres})

# Define the feature extraction functions
def extract_features_knn(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    features = []
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.rms(y=y)))

    return features

def extract_features_mfcc(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)
    return mfccs_mean

def extract_features_svm(audio_file_path):
    y, sr = librosa.load(audio_file_path, sr=None)
    features = []
    features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
    features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features.append(np.mean(librosa.feature.rms(y=y)))
    features.append(np.mean(librosa.feature.mfcc(y=y, sr=sr)))

    return features

# Extract the features from the audio files
knn_features = []
mfcc_features = []
svm_features = []
for audio_file in audio_files:
    knn_features.append(extract_features_knn(audio_file))
    mfcc_features.append(extract_features_mfcc(audio_file))
    svm_features.append(extract_features_svm(audio_file))

# Encode the genre labels as integers
label_encoder = LabelEncoder()
encoded_genres = label_encoder.fit_transform(genres)

# Split the data into training and testing sets for KNN
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(knn_features, encoded_genres, test_size=0.2, random_state=42)

# Split the data into training and testing sets for MFCC
X_train_mfcc, X_test_mfcc, y_train_mfcc, y_test_mfcc = train_test_split(mfcc_features, encoded_genres, test_size=0.2, random_state=42)

# Split the data into training and testing sets for SVM
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X, y, test_size=0.2, random_state=42)

# Encode the genre labels for SVM
encoded_genres_svm = label_encoder.transform(df['label'].values)

# Train the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_knn, y_train_knn)

# Test the KNN model
y_pred_knn = knn_model.predict(X_test_knn)
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
print('KNN accuracy:', accuracy_knn)

# Train the MFCC model
mfcc_model = KNeighborsClassifier(n_neighbors=5)
mfcc_model.fit(X_train_mfcc, y_train_mfcc)

# Test the MFCC model
y_pred_mfcc = mfcc_model.predict(X_test_mfcc)
accuracy_mfcc = accuracy_score(y_test_mfcc, y_pred_mfcc)
print('MFCC accuracy:', accuracy_mfcc)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_svm, y_train_svm)

# Test the SVM model
y_pred_svm = svm_model.predict(X_test_svm)
accuracy_svm = accuracy_score(y_test_svm, y_pred_svm)
print('SVM accuracy:', accuracy_svm)

# Specify the path to the file you want to test
test_file = 'blues.00000.wav'
test_file_path = os.path.join(dataset_path, 'blues', test_file)

# Extract features from the test file
test_features_knn = extract_features_knn(test_file_path)
test_features_mfcc = extract_features_mfcc(test_file_path)
test_features_svm = extract_features_svm(test_file_path)

# Reshape the test features for KNN and MFCC models
test_features_knn = np.array(test_features_knn).reshape(1, -1)
test_features_mfcc = np.array(test_features_mfcc).reshape(1, -1)

# Predict the genre using the KNN model
knn_prediction = label_encoder.inverse_transform(knn_model.predict(test_features_knn))
print('KNN prediction:', knn_prediction[0])

# Predict the genre using the MFCC model
mfcc_prediction = label_encoder.inverse_transform(mfcc_model.predict(test_features_mfcc))
print('MFCC prediction:', mfcc_prediction[0])

# Predict the genre using the SVM model
svm_features = np.array(test_features_svm).reshape(1, -1)
svm_prediction = svm_model.predict(svm_features)
svm_prediction = label_encoder.inverse_transform(svm_prediction)
print('SVM prediction:', svm_prediction[0])