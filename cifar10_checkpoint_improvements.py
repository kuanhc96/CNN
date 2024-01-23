from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.miniVGGNet import MiniVGGNet
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="path to weights directory")
args = vars(ap.parse_args())

((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

optimizer = SGD(learning_rate=0.01, weight_decay=0.01/40, momentum=0.9, nesterov=True)
model = MiniVGGNet.build(width=32, height=32, depth=3, num_classes=10)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# construct callback used to save best models
# `fname`: this is a template string that will be interpreted by the callback function
# `epoch:03d` denotes the "epoch" number, written out to 3 digits
# `val_loss:.4f` denotes the validation loss, written out to 4 significant digits (f means float)
fname = os.path.sep.join([args["weights"], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])

# `mode`="min" denotes that the ModelCheckpoint should determinte "best" by "lower the better"
# when monitoring "val_loss", mode="min" should be used
# when monitorying "val_acc", mode="max" should be used (higher the better)
# if only one weight file is needed (the best one), simply replace `fname` with the output path, `args["weights"]`
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

training_history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)