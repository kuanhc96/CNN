from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import backend as K

class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, num_classes):
        model = Sequential()
        input_shape = (height, width, depth)
        channel_dim = -1 # the dimension of the channel is the "last" item

        if K.image_data_format() == "channel_first":
            input_shape = (depth, height, width)
            channel_dim = 1 # the dimension of the channel is the "first" item
            # the documentation of BatchNormalization: 
            # - axis: Integer, the axis that should be normalized (typically the features axis). For instance, after a Conv2D layer with data_format="channels_first", use axis=1.
            

        # CONV => ACT => BN
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=input_shape))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=channel_dim))

        # CONV => ACT => BN => POOL => DROPOUT
        model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # CONV => ACT => BN
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=channel_dim))

        # CONV => ACT => BN => POOL => DROPOUT
        model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=channel_dim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # Fully connected => ACT => BN => DROPOUT
        model.add(Flatten())
        model.add(Dense(units=512))
        model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=channel_dim))
        model.add(Dropout(0.5))
        # Flatten is not needed, since the inputs were already flattened

        # Fully connected => ACT => BN => DROPOUT
        model.add(Dense(units=num_classes))
        model.add(Activation("softmax"))

        return model