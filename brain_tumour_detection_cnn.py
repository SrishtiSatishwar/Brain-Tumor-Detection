!pip install tensorflow
!pip install --upgrade tensorflow
!pip install tf-nightly

!pip install keras

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Step 1: Assign label 1 to tumor images
tumor_folder = "/content/yes-tumor-meningioma-3"
tumor_images = []
tumor_labels = []
for filename in os.listdir(tumor_folder):
    img = load_img(os.path.join(tumor_folder, filename))
    img_array = img_to_array(img)
    tumor_images.append(img_array)
    tumor_labels.append(1)

# Step 2: Append tumor images and labels into images and labels arrays
images = tumor_images
labels = tumor_labels

# Step 3: Assign label 0 to non-tumor images
non_tumor_folder = "/content/no-tumor3"
non_tumor_images = []
non_tumor_labels = []
for filename in os.listdir(non_tumor_folder):
    img = load_img(os.path.join(non_tumor_folder, filename))
    img_array = img_to_array(img)
    non_tumor_images.append(img_array)
    non_tumor_labels.append(0)

# Step 4: Append non-tumor images and labels into images and labels arrays
images += non_tumor_images
labels += non_tumor_labels

# Step 5: Shuffle and resize images
random.seed(42)
combined = list(zip(images, labels))
random.shuffle(combined)
images, labels = zip(*combined)
images = [tf.image.resize(image, (64, 64)) / 255.0 for image in images]

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Step 7: Create train and test generators
train_generator = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_generator = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Step 8: Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Step 9: Fit the model on the train generator
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_generator.batch(32), epochs=40)

# Step 10: Predict on the test generator
predictions = model.predict(test_generator.batch(32))
y_pred = np.round(predictions).flatten()

# EVALUATION METHOD 1
# Step 11: Evaluate the model on various metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, predictions)
confusion = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)

# EVALUATION METHOD 2 - results are the same

# Convert test generator to NumPy arrays
X_test_np = np.array(list(test_generator.map(lambda x, y: x)))
y_test_np = np.array(list(test_generator.map(lambda x, y: y)))

# Predict on the test set
predictions = model.predict(X_test_np)
y_pred = np.round(predictions).flatten()

# Calculate evaluation metrics
accuracy = accuracy_score(y_test_np, y_pred)
precision = precision_score(y_test_np, y_pred)
recall = recall_score(y_test_np, y_pred)
f1 = f1_score(y_test_np, y_pred)
roc_auc = roc_auc_score(y_test_np, predictions)
confusion = confusion_matrix(y_test_np, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("ROC AUC:", roc_auc)

fpr, tpr, thresholds = roc_curve(y_test_np, predictions)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.show()
