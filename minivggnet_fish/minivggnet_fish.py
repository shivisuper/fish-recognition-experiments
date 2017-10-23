# USAGE
# python minivggnet_fish.py --dataset
# /mnt/dissertation_work/final_dataset --output output/fish_minivggnet_with_bn.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from research.preprocessing import AspectAwarePreprocessor
from research.preprocessing import ImageToArrayPreprocessor
from research.datasets import SimpleDatasetLoader
from research.nn.conv import MiniVGGNet
from research.nn.conv import MiniVGGNetNoBN
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
import numpy as np
import argparse
from imutils import paths
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="path to input database")
ap.add_argument("-o", "--output", required=True,
                help="path to the output loss/accuracy plot")
ap.add_argument("-b", "--bn", type=int,
                required=True, help="whether require bn or not")
args = vars(ap.parse_args())

Network = MiniVGGNet if args["bn"] else MiniVGGNetNoBN

# load the training and testing data, then scale it into the
# range [0, 1]
print("[INFO] loading images from 10-fish dataset...")
imagePaths = list(paths.list_images(args["dataset"]))
classNames = os.listdir(args["dataset"])

#initialize image processors
aap = AspectAwarePreprocessor(64,64)
iap = ImageToArrayPreprocessor()

#load the dataset from disk and then scale the raw pixel intensities
#to the range [0,1]
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])
(data, labels) = sdl.load(imagePaths, verbose=-1)
data = data.astype("float") / 255.0

(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25,
                                                  random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 20, momentum=0.9, nesterov=True)
model = Network.build(width=64, height=64, depth=3, classes=len(classNames))
model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
        batch_size=64, epochs=20, verbose=1)

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1), target_names=classNames))

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Fish-10 database")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
