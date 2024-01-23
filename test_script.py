from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
import cv2

# numpy arrays are suitable formats for cv2 related functions and displays
img_data = np.random.random(size=(100, 100, 3))
print(img_data.shape)
print(type(img_data))
cv2.imshow("img_data", img_data)
cv2.waitKey()


# numpy arrays can be converted to PIL images using tensorflow's array_to_img function
img_data = array_to_img(img_data)
print(type(img_data))
img_data.show()

# On the other hand, PIL images can be converted back to numpy arrays if cv2 functions are needed
array_image_data = img_to_array(img_data, data_format=None)
print(array_image_data.shape)
print(type(array_image_data))