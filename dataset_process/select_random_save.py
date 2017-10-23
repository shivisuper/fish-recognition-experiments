import random as r
from sys import exit
from os import path, makedirs, walk
from imutils import paths
from shutil import copy2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
                help='Path to input directory')
ap.add_argument('-o', '--output', required=True,
                help='Path to output directory')
ap.add_argument('-r', '--randomNumber', required=True, default=5,
                type=int, help='Number of random files to copy')
args = vars(ap.parse_args())

images_path = args["input"]
save_dir = args["output"]
num = args["randomNumber"]

if not path.exists(args["input"]):
    print("{}: No such directory".format(args["input"]))
    exit(0)

for dir_name, sub_dir, file_list in walk(images_path):
    if len(sub_dir) > 0:
        continue
    else:
        curr_dir = path.join(save_dir, path.basename(dir_name))
        if not path.exists(curr_dir):
            print("Creating directory '{}'".format(curr_dir))
            makedirs(curr_dir)
        random_imgs = []
        for i in range(0, num):
            file_path = path.join(dir_name, r.choice(file_list))
            random_imgs.append(file_path)
            print("Choosing random image #{}".format(i+1), end='\r')
        for i, img in enumerate(random_imgs):
            copy2(img, path.join(curr_dir, path.basename(img)))
            print("Copying random image #{}".format(i+1), end='\r')
        print()
