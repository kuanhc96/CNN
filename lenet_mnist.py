from pyimagesearch.nn.conv.lenet import LeNet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

((trainX, trainY), (testX, testY)) = mnist.load_data()

if K.image_data_format() == "channel_first":
    trainX = trainX.reshape((trainX.shape[0], 1, 28, 28))
    testX = testX.reshape((testX.shape[0], 1, 28, 28))

else:
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

# normalize data to range [0, 1]
# trainX = trainX.astype("float32") / 255.0
# testX = testX.astype("float32") / 255.0

le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)

optimizer = SGD(learning_rate = 0.01)
model = LeNet.build(width=28, height=28, depth=1, num_classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

training_history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=20, verbose=1)

predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), training_history.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), training_history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), training_history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 20), training_history.history["val_accuracy"], label="val_acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch #")
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()

model.save("lenet_batchsize64_epochs20_data_not_normalized.hdf5")