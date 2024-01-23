import numpy as np

class Perceptron:
    # N: number of features for an input data point
    # alpha: learning rate; default = 0.1
    def __init__(self, N, alpha=0.1):
        # initialize weight matrix and learning rate alpha
        self.W = np.random.randn(N+1) / np.sqrt(N) # N + 1 to account for the bias term
        self.alpha = alpha

    # step function
    def step(self, x):
        return 1 if x > 0 else 0
    
    def fit(self, X, y, iterations=10):
        # insert a column of ones to account for the bias term in the weight matrix
        X = np.c_[X, np.ones(X.shape[0])]
        for iteration in np.arange(0, iterations):
            # implement SGD with batch size = 1
            for (x, target) in zip(X, y): # loop over each individual data point, row by row
                # get dot product between the single data point x and the weight W,
                # and pass it to the step function
                p = self.step(np.dot(x, self.W))
                
                # only update weight matrix if the prediction does not match the target
                if p != target:
                    error = p - target
                    
                    # update weight matrix
                    self.W = self.W - self.alpha * error * x
    
    def predict(self, X, addBias=True):
        # ensure that the input is a matrix:
        # np.atleast_2d converts its input into a 2-D array
        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones(X.shape[0])]

        return self.step(np.dot(X, self.W))