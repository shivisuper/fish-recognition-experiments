# USAGE
# python finetune_fish10.py --model fish10.model --network vgg16

# import the necessary packages
#from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from research.preprocessing import ImageToArrayPreprocessor
from research.preprocessing import AspectAwarePreprocessor
from research.preprocessing import MeanPreprocessor
from research.callbacks import TrainingMonitor
#from research.datasets import SimpleDatasetLoader
from research.nn.conv import FCHeadNet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from research.io import HDF5DatasetGenerator
from keras.applications import VGG16
from keras.applications import ResNet50
from keras.applications import VGG19
from keras.applications import InceptionV3
from keras.layers import Input
from keras.models import Model
#from imutils import paths
from config import fish_config as config
import numpy as np
import argparse
import os
import json

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
        help="path to output model")
ap.add_argument("-n", "--network", required=True,
        help="pre-trained network to use")
args = vars(ap.parse_args())

MODEL = {
    "vgg16"     : VGG16,
    "vgg19"     : VGG19,
    "resnet"    : ResNet50,
    "inception" : InceptionV3
}

image_shape = (224, 224)

if args["network"] == "inception":
    image_shape = (299, 299)

Network = MODEL[args["network"]]

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=20, width_shift_range=0.15,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")

means = json.loads(open(config.DATASET_MEAN).read())

# initialize the image preprocessors
iap = ImageToArrayPreprocessor()
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
#pp = PatchPreprocessor(224, 224)

#construct set of callbacks
path = os.path.sep.join([config.OUTPUT_PATH, "{}.png".format(
    os.getpid())])
callbacks = [TrainingMonitor(path)]

trainGen = HDF5DatasetGenerator(config.TRAIN_HDF5, 32, aug=aug,
                                preprocessors=[mp, iap], classes=10)
valGen = HDF5DatasetGenerator(config.VAL_HDF5, 32,
                              preprocessors=[mp, iap], classes=10)

# load the network, ensuring the head FC layer sets are left
# off
baseModel = Network(weights="imagenet", include_top=False,
                    input_tensor=Input(shape=(image_shape[0], image_shape[1], 3)))

# initialize the new head of the network, a set of FC layers
# followed by a softmax classifier
headModel = FCHeadNet.build(baseModel, 10 , 256)

# place the head FC model on top of the base model -- this will
# become the actual model we will train
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they
# will *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

# compile our model (this needs to be done after our setting our
# layers to being non-trainable
print("[INFO] compiling model...")
opt = RMSprop(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the head of the network for a few epochs (all other
# layers are frozen) -- this will allow the new FC layers to
# start to become initialized with actual "learned" values
# versus pure random
print("[INFO] training head...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 128,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 128,
    epochs=25,
    #max_queue_size=128*2,
    callbacks=callbacks, verbose=1)

#model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
        #validation_data=(testX, testY), epochs=25,
        #steps_per_epoch=len(trainX) // 32, verbose=1)

# evaluate the network after initialization
print("[INFO] evaluating after initialization...")
predictions = model.predict_generator(valGen.generator(),
                                      steps=valGen.numImages // 128,
                                      max_queue_size=128 * 2)
# Get most likely classes
predicted_classes = np.argmax(predictions, axis=1)
# get ground truth classes and class-labels
true_classes = valGen.classes
class_labels = os.listdir(config.IMAGES_PATH)

# get a classification report
print(classification_report(true_classes, predicted_classes,
                            target_names=class_labels))

#print(classification_report(testY.argmax(axis=1),
        #predictions.argmax(axis=1), target_names=classNames))

# now that the head FC layers have been trained/initialized, lets
# unfreeze the final set of CONV layers and make them trainable
for layer in baseModel.layers[15:]:
        layer.trainable = True

# for the changes to the model to take affect we need to recompile
# the model, this time using SGD with a *very* small learning rate
print("[INFO] re-compiling model...")
opt = SGD(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy"])

# train the model again, this time fine-tuning *both* the final set
# of CONV layers along with our set of FC layers
print("[INFO] fine-tuning model...")
model.fit_generator(
    trainGen.generator(),
    steps_per_epoch=trainGen.numImages // 128,
    validation_data=valGen.generator(),
    validation_steps=valGen.numImages // 128,
    epochs=50,
    #max_queue_size=128*2,
    callbacks=callbacks, verbose=1)

#model.fit_generator(aug.flow(trainX, trainY, batch_size=32),
        #validation_data=(testX, testY), epochs=100,
        #steps_per_epoch=len(trainX) // 32, verbose=1)

# evaluate the network on the fine-tuned model
print("[INFO] evaluating after fine-tuning...")
predictions = model.predict_generator(valGen.generator(),
                                      steps=valGen.numImages // 128,
                                      max_queue_size=128 * 2)

predicted_classes = np.argmax(predictions, axis=1)
# get a classification report
print(classification_report(true_classes, predicted_classes,
                            target_names=class_labels))

#predictions = model.predict(testX, batch_size=32)
#print(classification_report(testY.argmax(axis=1),
        #predictions.argmax(axis=1), target_names=classNames))

# save the model to disk
print("[INFO] serializing model...")
model.save(args["model"])

trainGen.close()
valGen.close()
