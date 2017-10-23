# organize imports
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import _pickle
import h5py
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# load the user configs
with open('conf.json') as f:
        config = json.load(f)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",
                default="/mnt/dissertation_work/resizedfish/resized224x224AA",
                help="Path to the training dataset")
ap.add_argument("-m", "--model",
                default="vgg16", help="Name of pre-trained network to")
ap.add_argument("-o", "--output",
                default="output/fish10", help="Output path for features")
args = vars(ap.parse_args())

final_dir = os.path.join(args["output"], args["model"])
if not os.path.exists(final_dir):
    os.makedirs(final_dir)

if not os.path.exists(os.path.join(final_dir, "model")):
    os.makedirs(os.path.join(final_dir, "model"))

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
seed                    = config["seed"]
classifier_path         = config["classifier_path"]

# import features and labels
h5f_data = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print("[INFO] features shape: {}".format(features.shape))
print("[INFO] labels shape: {}".format(labels.shape))

print("[INFO] training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)

print("[INFO] splitted train and test data...")
print("[INFO] train data  : {}".format(trainData.shape))
print("[INFO] test data   : {}".format(testData.shape))
print("[INFO] train labels: {}".format(trainLabels.shape))
print("[INFO] test labels : {}".format(testLabels.shape))

# use logistic regression as the model
print("[INFO] creating model...")
model = LogisticRegression(random_state=seed)
model.fit(trainData, trainLabels)

# use rank-1 and rank-5 predictions
print("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0
rank_5 = 0

# loop over test data
for (label, features) in zip(testLabels, testData):
        # predict the probability of each class label and
        # take the top-5 class labels
        predictions = model.predict_proba(np.atleast_2d(features))[0]
        predictions = np.argsort(predictions)[::-1][:5]

        # rank-1 prediction increment
        if label == predictions[0]:
                rank_1 += 1

        # rank-5 prediction increment
        if label in predictions:
                rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100

# write the accuracies to file
f.write("rank-1: {:.2f}%\n".format(rank_1))
f.write("rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData)

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print("[INFO] dumping classifier...")
f = open(classifier_path, "wb")
f.write(_pickle.dumps(model))
f.close()

# display the confusion matrix
print ("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm,
            annot=True,
            cmap="Set2")
plt.show()
