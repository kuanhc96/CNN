from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import cifar10
import cv2
import numpy as np

((trainX, trainY), (testX, testY)) = cifar10.load_data()
random_photo = np.array([testX[0].astype("float") / 255.0])

model = load_model("./cifar10_weights_32kernels_40epochs.hdf5")

prediction = model.predict(random_photo)
label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
print(label_names[prediction.argmax(axis=1)[-1]])