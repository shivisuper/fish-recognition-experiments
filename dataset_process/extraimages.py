from imutils import paths
import argparse
from os import path, makedirs, getcwd
from shutil import copyfile

ap = argparse.ArgumentParser()
ap.add_argument("-p1", "--path1", required=True,
                help="Path to original image")
ap.add_argument("-p2", "--path2", help="Path to existing image",
                default="/home/shivisuper/Downloads/final_dataset")
ap.add_argument("-n", "--number", help="Number of images to select",
                type=int, default=5)
ap.add_argument("-s", "--save", help="Path where to save images",
                default=path.join(getcwd(), "extra_dataset"))
args = vars(ap.parse_args())

if not path.exists(args["save"]):
    makedirs(args["save"])

num = args["number"]

original_images = list(paths.list_images(args["path1"]))
original_images_nopath = [path.split(img)[1] for img in original_images]

existing_folder = path.split(args["path1"])[1] + "_every_10"
new_folder_path = path.join(args["save"], path.split(args["path1"])[1] + "_every_" + str(num))

if not path.exists(new_folder_path):
    makedirs(new_folder_path)

existing_images = list(paths.list_images(path.join(args["path2"], existing_folder)))
existing_images_nopath = [path.split(img)[1] for img in existing_images]

new_nopath = set(original_images_nopath) - set(existing_images_nopath)
new_images = [path.join(args["path1"], img) for img in new_nopath]

list_of_images = new_images[0::num]

print("Started copying...")
for (i, item) in enumerate(list_of_images):
    filename = path.split(item)[1]
    copyfile(item, path.join(new_folder_path, filename))
    print("Copied {}/{}".format(i+1, len(list_of_images)), end="\r")
print("Finished copying!")