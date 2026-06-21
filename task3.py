import os
import cv2
import numpy as np

# Dataset paths
cat_path = "PetImages/Cat"
dog_path = "PetImages/Dog"

X = []
y = []

# Load cat images
for image_name in os.listdir(cat_path)[:500]:
    img_path = os.path.join(cat_path, image_name)
    img = cv2.imread(img_path)

    if img is not None:
        try:
            img = cv2.resize(img, (64, 64))
            img = img.flatten()
            img = img / 255.0
            X.append(img)
            y.append(0)  # Cat
        except:
            pass

# Load dog images
for image_name in os.listdir(dog_path)[:500]:
    img_path = os.path.join(dog_path, image_name)
    img = cv2.imread(img_path)

    if img is not None:
        try:
            img = cv2.resize(img, (64, 64))
            img = img.flatten()
            img = img / 255.0
            X.append(img)
            y.append(1)  # Dog
        except:
            pass
# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("Total Images Loaded:", len(X))
# print("Feature Shape:", X.shape)

# Split dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

# Create SVM model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
svm_model = SVC(kernel="linear")
print("Training Model...")
svm_model.fit(X_train, y_train)

# Prediction
y_pred = svm_model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Classification Report
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Predict a new image
img_path = input("\nEnter image path: ")
img = cv2.imread(img_path)

if img is not None:
    img = cv2.resize(img, (64, 64))
    img = img.flatten()
    img = img / 255.0
    img = img.reshape(1, -1)

    prediction = svm_model.predict(img)

    if prediction[0] == 0:
        print("Prediction: Cat 🐱")
    else:
        print("Prediction: Dog 🐶")
else:
    print("Invalid image path.")       