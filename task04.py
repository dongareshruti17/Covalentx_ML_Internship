import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

# Dataset path
dataset_path = "leapGestRecog"
X = []
y = []

# Use only these 5 gestures
selected_gestures = ["01_palm","03_fist","05_thumb","06_index","07_ok" ]

# Folder name to readable name
gesture_names = { "01_palm": "Palm","03_fist": "Fist","05_thumb": "Thumbs Up",
    "06_index": "Index Finger","07_ok": "OK Sign" }

print("Loading Images...")

# Read images
for person in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person)
    if os.path.isdir(person_path):
        for gesture in selected_gestures:
            gesture_path = os.path.join(person_path,gesture)
            if os.path.exists(gesture_path):
                for image_name in os.listdir(gesture_path):
                    image_path = os.path.join(gesture_path,image_name)
                    img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img,(64, 64))
                        X.append(img)
                        y.append(gesture)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

print("Total Images Loaded:", len(X))
# Normalize images
X = X / 255.0

# Reshape for CNN
X = X.reshape(-1, 64, 64, 1)

# Convert labels to numbers
classes = np.unique(y)

label_map = {}
for i, c in enumerate(classes):
    label_map[c] = i
y = np.array([label_map[label] for label in y])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

# Build CNN model
model = Sequential()
model.add(Conv2D(32,(3, 3),activation="relu",input_shape=(64, 64, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64,(3, 3),activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(len(classes),activation="softmax"))

# Compile model
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

print("\nTraining Model...")
# Train model
history = model.fit(X_train,y_train,epochs=5,batch_size=32,validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test,y_test)
print("\nModel Accuracy:",round(accuracy * 100, 2),"%")

# Predict one random image
index = np.random.randint(0,len(X_test))
image = X_test[index]

prediction = model.predict(image.reshape(1, 64, 64, 1))
predicted_class = np.argmax(prediction)
folder_name = classes[predicted_class]
gesture = gesture_names[folder_name]
print("\nPredicted Gesture:", gesture)

# Display image
plt.imshow(image.reshape(64, 64),cmap="gray")
plt.title("Predicted Gesture: " + gesture)
plt.axis("off")
plt.show()