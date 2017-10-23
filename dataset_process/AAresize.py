from augment_data.research.preprocessing import AspectAwarePreprocessor
import os
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",
                default="/mnt/dissertation_work/fish_dataset_raw",
                help="Path to input image set")
ap.add_argument("-w", "--width", required=True,
                type=int, help="Width of resized image")
ap.add_argument("-he", "--height", required=True,
                type=int, help="height of resized image")
ap.add_argument("-o", "--output",
                default="/mnt/dissertation_work/resizedfish",
                help="Path where resized will be stored")
args = vars(ap.parse_args())
w = args["width"]
h = args["height"]

output_dir = os.path.join(args["output"], "resized" + \
                          str(w) + 'x' + str(h) + 'AA')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

processor = AspectAwarePreprocessor(w, h)

for root_dir, sub_dir, file_list in os.walk(args["input"]):
    if len(sub_dir) > 0:
        continue
    else:
        new_dir = os.path.join(output_dir,
                               os.path.basename(root_dir))
        print("Creating directory {}".format(new_dir))
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
        print("Resizing images...")
        for i, f in enumerate(file_list):
            img = cv2.imread(os.path.join(root_dir, f))
            resized = processor.preprocess(img)
            print("Resized {}/{}".format(i+1, len(file_list)),
                  end='\r')
            cv2.imwrite(os.path.join(new_dir, f), resized)
        print()
        print("Finished label {}".format(os.path.basename(root_dir)))
print("Finished process")

