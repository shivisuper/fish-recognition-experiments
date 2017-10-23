# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import Xception
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.models import model_from_json
import _pickle

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",
                default="/mnt/dissertation_work/resizedfish/resized224x224",
                help="Path to the training dataset")
ap.add_argument("-m", "--model",
                default="vgg16", help="Name of pre-trained network to")
ap.add_argument("-o", "--output",
                default="output/fish10", help="Output path for features")
args = vars(ap.parse_args())

MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50
}

# load the user configs
with open('config/conf.json') as f:
        config = json.load(f)

final_dir = os.path.join(args["output"], args["model"])
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

if not os.path.exists(os.path.join(final_dir, "model")):
    os.makedirs(os.path.join(final_dir, "model"))

# config variables
#model_name              = config["model"]
model_name              = args["model"]
weights                 = config["weights"]
include_top             = config["include_top"]
#train_path              = config["train_path"]
train_path              = args["input"]
#features_path           = config["features_path"]
features_path           = os.path.join(final_dir, "features.h5")
#labels_path             = config["labels_path"]
labels_path             = os.path.join(final_dir, "labels.h5")
test_size               = config["test_size"]
#results                 = config["results"]
results                 = os.path.join(final_dir, "results.txt")

# start time
print("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

Network = MODELS[args["model"]]

image_size = (224, 224)
preprocess = imagenet_utils.preprocess_input

if args["model"] in ("inception", "xception"):
    image_size = (299, 299)
    preprocess = preprocess_input

if args["model"] in ("vgg19", "vgg16"):
    layer = "fc1"
elif args["model"] in ("resnet", "inception"):
    layer = "flatten"
else:
    layer = "avg_pool"

# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
print("Loading model {}".format(args["model"]))

model = Network(weights=weights)
#model = Model(input=base_model.input, output=base_model.get_layer(layer).output)

print("[INFO] successfully loaded base model and model...")

# path to training dataset
train_labels = os.listdir(train_path)

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder
for label in train_labels:
        cur_path = train_path + "/" + label
        i = 0
        for image_path in glob.glob(cur_path + "/*.jpg"):
                img = image.load_img(image_path)
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess(x)
                feature = model.predict(x)
                flat = feature.flatten()
                features.append(flat)
                labels.append(label)
                i += 1
                print("[INFO] processed - {}".format(i), end='\r')
        print()
        print("[INFO] completed label - {}".format(label))

# encode the labels using LabelEncoder
targetNames = np.unique(labels)
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print("[STATUS] training labels: {}".format(le_labels))
print("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

print("[STATUS] features and labels saved..")

# end time
end = time.time()
print("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
