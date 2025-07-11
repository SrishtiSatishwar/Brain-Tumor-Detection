import cv2
from sklearn import svm
from skimage.io import imread
from skimage.feature import hog
from skimage.color import rgb2gray
import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

tumor_path = "/content/yes-tumor-meningioma-3"
non_tumor_path = "/content/no-tumor-3"

X = []  # List to store the feature vectors
y = []  # List to store the labels (0 for non-tumor, 1 for tumor)

# Process the tumor images
for filename in os.listdir(tumor_path):
    img_path = os.path.join(tumor_path, filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))  # Resize the image to a fixed size
    features = img.flatten()  # Convert the image to a 1D array
    X.append(features) # feature extraction for yes-tumor images and appending it to X array
    y.append(1)  # Assign label 1 for tumor images and appening it to Y array

# Process the non-tumor images
for filename in os.listdir(non_tumor_path):
    img_path = os.path.join(non_tumor_path, filename)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (64, 64))  # Resize the image to a fixed size
    features = img.flatten()  # Convert the image to a 1D array
    X.append(features) # feature extraction for no-tumor images and appending it to X array
    y.append(0)  # Assign label 0 for non-tumor images and appening it to Y array

# Convert the lists to arrays
X = np.array(X)
y = np.array(y)

# Split dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an SVM classifier
svm_classifier = svm.SVC()

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Evaluate the model

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC Score:", roc_auc)

# Calculate the probability scores for the positive class (tumor)
y_prob = svm_classifier.decision_function(X_test)

# Compute the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prob)

# Compute the AUC score
auc_score = metrics.auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
