# USAGE
# python inspect_model.py --include-top -1

# import the necessary packages
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
from keras.applications import InceptionV3
from shallownet.research.nn.conv import ShallowNet

import argparse

MODEL = {
    "vgg16"     : VGG16,
    "vgg19"     : VGG19,
    "inception" : InceptionV3,
    "resnet"    : ResNet50,
    "shallow"   : ShallowNet
}

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--include-top", type=int, default=1,
	help="whether or not to include top of CNN")
ap.add_argument("-m", "--model", required=True,
	help="name of the model")
args = vars(ap.parse_args())

include = args["include_top"] > 0

Network = MODEL[args["model"]]

# load the network
print("[INFO] loading network...")
model = Network(weights="imagenet",
	include_top=include)
print("[INFO] showing layers...")

# loop over the layers in the network and display them to the
# console
for (i, layer) in enumerate(model.layers):
	print("[INFO] {}\t{}".format(i, layer))
