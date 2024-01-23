from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.image_to_array_preprocessor import ImageToArrayPreprocessor
from pyimagesearch.preprocessing.simple_preprocessor import SimplePreprocessor
from pyimagesearch.datasets.simple_dataset_loader import SimpleDatasetLoader
from pyimagesearch.nn.conv.ShallowNet import ShallowNet
from tensorflow.keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", type=str, required=True, help="path to input dataset")
args = vars(ap.parse_args())
image_paths = list(paths.list_images(args["dataset"]))

simple_preprocessor = SimplePreprocessor(32, 32)
image_to_array_preprocessor = ImageToArrayPreprocessor()

# load the data
simple_dataset_loader = SimpleDatasetLoader(preprocessors=[simple_preprocessor, image_to_array_preprocessor])
(data, labels) = simple_dataset_loader.load(imagePaths=image_paths, verbose=500)

# rescale the images to [0, 1]
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# one hot encoding
label_binarizer = LabelBinarizer()
trainY = label_binarizer.fit_transform(trainY)
testY = label_binarizer.fit_transform(testY)

optimizer = SGD(lr=0.05)
model = ShallowNet(width=32, height=32, depth=3, num_classes=3).get_model()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# train model
EPOCHS = 100
train_history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=32, epochs=EPOCHS, verbose=1)

# test model
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=["cat", "dog", "panda"]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), train_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), train_history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), train_history.history["accuracy"], label="accuracy")
plt.plot(np.arange(0, EPOCHS), train_history.history["val_accuracy"], label="val_accuracy")
plt.title("training loss and accuracy")
plt.xlabel("# of epochs")
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()