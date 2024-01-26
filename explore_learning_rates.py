import matplotlib
matplotlib.use("Agg")

from pyimagesearch.learning_rate_schedulers import StepDecay, PolynomialDecay
from pyimagesearch.nn.conv.miniVGGNet import MiniVGGNet
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD 
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--epochs", type=int, default=100, help="# of epochs to for training")
ap.add_argument("-l", "--lr-output", type=str, default="./lr-output", help="output path to store learning rate plots")
ap.add_argument("-t", "--train-output", type=str, default="./training_result_diagrams", help="output path to store training result diagrams")
args = vars(ap.parse_args())

learning_rate_strategies = [None]
learning_rate_strategy_names = ["no decay", "step-factor=0.5", "step-factor=0.25", "linear", "polynomial=5"]
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

strategy = StepDecay(learning_rate=0.1, factor=0.5, drop_interval=15)
learning_rate_strategies.append(strategy)

strategy = StepDecay(learning_rate=0.1, factor=0.25, drop_interval=15)
learning_rate_strategies.append(strategy)

strategy = PolynomialDecay(max_epochs=args["epochs"], initial_learning_rate=0.1, power=1)
learning_rate_strategies.append(strategy)

strategy = PolynomialDecay(max_epochs=args["epochs"], initial_learning_rate=0.1, power=5)
learning_rate_strategies.append(strategy)

for current_strategy, current_strategy_name in zip( learning_rate_strategies, learning_rate_strategy_names ):

    optimizer = None
    if current_strategy_name == "standard":
        callbacks = []
        optimizer = SGD(learning_rate = 0.1, momentum=0.9, weight_decay=0.0 / args["epochs"])
    else:
        callbacks = [LearningRateScheduler(current_strategy)]
        optimizer = SGD(learning_rate = 0.1, momentum=0.9, weight_decay=0.0)

    model = MiniVGGNet.build(32, 32, 3, 10)
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

    training_history = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=128, epochs=args["epochs"], callbacks=callbacks, verbose=1)
    predictions = model.predict(testX, batch_size=128)
    print(f"Current strategy: {current_strategy_name}")
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=label_names))

    N = np.arange(0, args["epochs"])
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(N, training_history.history["loss"], label="train_loss")
    plt.plot(N, training_history.history["val_loss"], label="val_loss")
    plt.plot(N, training_history.history["accuracy"], label="train_acc")
    plt.plot(N, training_history.history["val_accuracy"], label="val_acc")
    plt.title(f"Training loss and accuracy on CIFAR10 using the {current_strategy_name} decay strategy")
    plt.xlabel("Epoch #")
    plt.ylabel("loss/accuracy")
    plt.legend()
    plt.savefig(os.path.join(args["train-output"], current_strategy_name, ".png"))
    
    strategy.plot(N, title=current_strategy_name)
    plt.savefig(os.path.join(args["lr-output"], current_strategy_name, ".png"))