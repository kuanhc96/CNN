# Xception is only compatible with tensorflow
from tensorflow.keras.applications import ResNet50, InceptionV3, Xception, VGG16, VGG19, imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import argparse 
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-m", "--model", type=str, default="vgg16", help="name of pre-trained network to use")
args = vars(ap.parse_args())

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

if args["model"] not in MODELS.keys():
    raise AssertionError("the model inputted is unavailable")

input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception", "xception"):
    input_shape = (299, 299)
    preprocess = preprocess_input

network = MODELS[args["model"]]
model = network(weights="imagenet")

image = load_img(args["image"], target_size=input_shape)
print(type(image))
print(image)
image = img_to_array(image)
print(type(image))
print(image.shape)

# currently, the images are represented in (width x height x depth)
# since batches of images are trained at a time, the 3-d format is invalid for the models
# as such, np.expand_dims is used so that the data is represented as a numpy array of shape
# (num_images x widht x height x depth)
# for instance, a single image prepared for the VGG16 model will end up having a shape of (1, 224, 224, 3)
image = np.expand_dims(image, axis=0)
# different models require different preprocessing steps for images
# as such, before prediction, the input image needs to be preprocessed
image = preprocess(image)

predictions = model.predict(image)
print(predictions)
P = imagenet_utils.decode_predictions(predictions)
print(P)

for (i, (imagenetID, label, prob)) in enumerate(P[0]):
    print(f"{i + 1}. {label}: { prob * 100}%")

# load the image via OpenCV, draw the top prediction on the image,
# and display the image to our screen
orig = cv2.imread(args["image"])
(imagenetID, label, prob) = P[0][0]
cv2.putText(orig, "Label: {}, {:.2f}%".format(label, prob * 100),
	(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("Classification", orig)
cv2.waitKey(0)