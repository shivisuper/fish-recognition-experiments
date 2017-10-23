from imutils import paths
import argparse
from os import path, makedirs, getcwd
from shutil import copyfile

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
                help="Path to image")
ap.add_argument("-n", "--number", required=True,
                help="Number of images to select")
ap.add_argument("-s", "--save", required=True,
                help="Path where to save images", default=getcwd())
args = vars(ap.parse_args())
num = int(args["number"])
list_of_images = list(paths.list_images(args["path"]))
n_list = list_of_images[0::num]
path_to_save = args["save"]
folder_to_save = path.split(args["path"])[1] + "_every_" + str(num)
if not path.exists(path.join(path_to_save, folder_to_save)):
    makedirs(path.join(path_to_save, folder_to_save))
for item in n_list:
    filename = path.split(item)[1]
    copyfile(item, path.join(path_to_save, folder_to_save, filename))