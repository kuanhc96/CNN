import matplotlib
matplotlib.use("Agg")

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.miniVGGNet import MiniVGGNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

EPOCHS=40
optimizer = SGD(learning_rate=0.01, weight_decay=0.01/EPOCHS, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, num_classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

training_history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=EPOCHS, verbose=1)

predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), training_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), training_history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), training_history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 40), training_history.history["val_accuracy"], label="val_acc")
plt.title("training loss and accuracy on CIFAR10")
plt.xlabel("epoch #")
plt.ylabel("loss/accuracy")
plt.legend()
plt.savefig(args["output"])
model.save("minivggnet_batchsize64_epochs40_decay_momentum_nesterov_no_normalization.hdf5")