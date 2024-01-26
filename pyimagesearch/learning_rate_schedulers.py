import matplotlib.pyplot as plt
import numpy as np

class LearningRateDecay:
    def plot(self, epochs, title="Learning Rate Schedule"):
        learning_rates = [self(i) for i in epochs] # self(i) calling the __call__ method of LearningRateDecay

        plt.style.use("ggplot")
        plt.figure()
        plt.plot(epochs, learning_rates)
        plt.title(title)
        plt.xlabel("Epoch #")
        plt.ylabel("Learning Rate")

class StepDecay(LearningRateDecay):
    def __init__(self, learning_rate=0.01, factor=0.5, drop_interval=10):
        self.learning_rate = learning_rate
        self.factor = factor
        self.drop_interval = drop_interval

    def __call__(self, epoch):
        exponent = np.floor(( epoch + 1 ) / self.drop_interval)
        return float(self.learning_rate * self.factor ** ( exponent )) 

class PolynomialDecay(LearningRateDecay):
    def __init__(self, max_epochs=100, initial_learning_rate=0.01, power=1.0):
        self.max_epochs=max_epochs
        self.initial_learning_rate = initial_learning_rate
        self.power = power

    def __call__(self, epoch):
        decay = (1 - (epoch / float(self.max_epochs))) ** self.power
        alpha = self.initial_learning_rate * decay

        return float(alpha)