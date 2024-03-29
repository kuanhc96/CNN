from tensorflow.keras.preprocessing.image import img_to_array
class ImageToArrayPreprocessor:
    def __init__(self, data_format=None):
        self.data_format = data_format

    def preprocess(self, image):
        # for notes on img_to_array, checkout `test_script.py`
        return img_to_array(image, data_format = self.data_format)