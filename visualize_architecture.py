import os
from keras.utils import plot_model
from keras.applications import VGG16
from shallownet.research.nn.conv import ShallowNet
from transfer_learning.research.nn.conv import FCHeadNet
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
                help="Name of the png file")
ap.add_argument("-m", "--model", required=True,
                help="Name of network")
args = vars(ap.parse_args())

MODEL = {
    "shallow": ShallowNet,
    "vgg16": VGG16,
    "fchead": FCHeadNet
}

Network = MODEL[args["model"]]

print("Building the model...")
model = Network.build(100, 100, 3, 10)
print("Plotting the model...")
plot_model(model, to_file=args["name"], show_shapes=True)
print("Plot {} created".format(args["name"]))
