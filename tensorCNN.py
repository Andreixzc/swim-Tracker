import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
import cv2

def preprocess_images(image_paths, target_size):
    images = []
    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
        image = cv2.resize(image, target_size)  
        image = image / 255.0 
        images.append(image)
    return np.array(images)

image_paths = [...]  
labels = [...]  

target_size = (32, 32) 

images = preprocess_images(image_paths, target_size)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(target_size[0], target_size[1], 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# test
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
