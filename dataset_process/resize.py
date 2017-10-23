from myutils import resize
import os
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", type=int, default=50,
                help="width of the resized image")
ap.add_argument("-he", "--height", type=int, default=50,
                help="height of the resized image")
ap.add_argument("-i", "--input", default="/mnt/dissertation_work/rawfish_dataset",
                help="input path")
args = vars(ap.parse_args())

rootDir = args["input"]
outputDir = '/mnt/dissertation_work/resizedfish/resized' + str(args["width"]) + 'x' + str(args["height"])

dim = (args["width"], args["height"])

for dirName, subDirs, fileList in os.walk(rootDir):
    # skip the root directory
    if len(subDirs) > 0:
        continue
    else:
        saveDir = os.path.join(outputDir, os.path.basename(dirName))
        if not os.path.exists(saveDir):
            os.makedirs(os.path.join(outputDir, os.path.basename(dirName)))
            print("Creating directory {}".format(saveDir))
        for i, img in enumerate(fileList):
            img_path = os.path.join(dirName, img)
            load_img = cv2.imread(img_path)
            resized_img = resize(load_img, dim[0], dim[1])
            cv2.imwrite(os.path.join(saveDir, os.path.basename(img)), resized_img)
            print("Resizing image {} of {}".format(i + 1, len(fileList)), end='\r')
        print()
        print("Finished with {}".format(os.path.basename(dirName)))
print("Resizing done")
