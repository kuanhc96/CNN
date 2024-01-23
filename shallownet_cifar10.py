from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from pyimagesearch.nn.conv.ShallowNet import ShallowNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse 

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", type=str, default="")
args = vars(ap.parse_args())

weights_path = args["weights"]

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


optimizer = SGD(learning_rate=0.01)
model = ShallowNet(width=32, height=32, depth=3, num_classes=len(label_names)).get_model()
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])


EPOCHS=40
train_history = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32, verbose=1)

if weights_path != "":
    model.save(weights_path)

predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

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