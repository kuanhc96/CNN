from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras import backend as K

class ShallowNet:
    def __init__(self, width, height, depth, num_classes):
        self.width = width
        self.height = height
        self.depth = depth
        self.num_classes = num_classes
        
        self.model = Sequential()
        self.input_shape = (self.width, self.height, self.depth)

        if K.image_data_format == "channels_first":
            self.input_shape = (self.depth, self.width, self.height)
        
        # convaolutional layer
        # There are 32 kernels in this layer, each of size (3, 3)
        self.model.add(Conv2D(32, (3, 3), padding="same", input_shape = self.input_shape))
        self.model.add(Activation("relu"))

        # Fully connected layer
        self.model.add(Flatten())
        self.model.add(Dense(self.num_classes))
        self.model.add(Activation("softmax"))

    def get_model(self):
        return self.model