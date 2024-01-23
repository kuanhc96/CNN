from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, num_classes):
        model = Sequential()
        input_shape = (height, width, depth)
        kernel_size = 5
        pool_size = 2
        stride = 2

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)

        # CONV -> RELU -> POOL
        model.add(Conv2D(20, (kernel_size, kernel_size), padding="same", input_shape=input_shape)) # input_shape field only needed for first layer
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(stride, stride)))

        # CONV -> RELU -> POOL
        model.add(Conv2D(50, (kernel_size, kernel_size), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(pool_size, pool_size), strides=(stride, stride)))

        # fully connected layer -> relu
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # final layer (classification)
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))

        return model