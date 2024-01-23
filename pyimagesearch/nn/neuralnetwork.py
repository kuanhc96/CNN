import numpy as np

class NeuralNetwork:
    # inputs:
    # layers: a list of integers representing the architecure of the NN. 
    # e.g., 3-3-1, for an NN with 3 input nodes, 3 nodes in the hidden layer, and 1 output node
    # Note that the values in the `layers` list does not include the bias term by default
    # alpha: learning rate
    def __init__(self, layers, alpha=0.1):
        # initialize list of weight matrices
        self.W = []
        self.layers = layers
        self.alpha = alpha

        # initialize the weights in the neural network except for the last 2 layers
        for i in np.arange(0, len(layers) - 2):
            # the weight matrix between layer[i] and layer[i+1] need to have (layer[i] x layer[i+1]) elements
            # However many weights leaving layer[i] should be the same as the number of nodes in layer[i];
            # However many weights entering layer[i+1] should be the same as the number of nodes entering layer[i+1]
            w = np.random.randn(layers[i] + 1, layers[i+1] + 1) # the +1 at the end is for the bias term
            self.W.append(w / np.sqrt(layers[i]))
        
        # The final layer (activation function) does NOT include a bias term, so the initialization is handled separately
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))
    
    def __repr__(self):
        return f"NeuralNetwork: {self.layers}"

    def sigmoid(self, x):
        # the activation function
        return 1.0 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # derived by calculus
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def fit(self, X, y, iterations=1000, display_update=100):
        X = np.c_[X, np.ones(X.shape[0])] 

        for iteration in np.arange(0, iterations):

            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            if iteration == 0 or (iteration + 1) % display_update == 0:
                loss = self.calculate_loss(X, y)
                print(f"iteration={iteration + 1}, loss={loss}")

    def fit_partial(self, x, y):
        # the first entry to `layer_output` is just the input to the first (0th) layer
        layer_output = [np.atleast_2d(x)]

        # feedforward:
        for layer in np.arange(0, len(self.W)):
            next_layer_input = layer_output[layer].dot(self.W[layer])
            next_layer_input = self.sigmoid(next_layer_input)
            layer_output.append(next_layer_input) # the next layer's input is the output of the current layer
        
        # backpropogation
        error = layer_output[-1] - y # the last entry to `layer_output` is the final result of feedforward, i.e., the prediction
        layer_delta = [error * self.sigmoid_derivative(layer_output[-2].dot(self.W[-1]))] # the first entry is the gradient of the final layer

        for layer in np.arange(len(layer_output) - 2, 0, -1): # start looping from the second to last layer
            current_delta = layer_delta[-1].dot(self.W[layer].T)
            current_delta = current_delta * self.sigmoid_derivative(layer_output[layer-1].dot(self.W[layer-1]))
            layer_delta.append(current_delta)

        layer_delta = layer_delta[::-1]

        for layer in range(0, len(self.W)):
            self.W[layer] = self.W[layer] - self.alpha * layer_output[layer].T.dot(layer_delta[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss