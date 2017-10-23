import cv2
import argparse
import os

# Playing video from file:
ap = argparse.ArgumentParser()
# ask for path where video resides
ap.add_argument("-i", "--inputPath", required=True, help="Path to the video")
# ask for path where frames will be extracted. By default it is the directory where this script executes
ap.add_argument("-o", "--outputPath", required=True, help="Path where frames will be extracted",
                default=os.getcwd())
args = vars(ap.parse_args())

video_path = args["inputPath"]

frames_folder = os.path.join(args["outputPath"], os.path.split(video_path)[1].split('.')[0])

# create an object that will seek through the video
cap = cv2.VideoCapture(video_path)

try:
    # create the folder for holding extracted frames if it doesn't exists
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
except OSError:
    print('Error: Creating directory of data')

currentFrame = 1
while True:
    # while we are able to read new frames, keep on extracting them
    (ret, frame) = cap.read()
    # ret is boolean telling whether we were able to extract a frame or not
    if ret:
        # Saves image of the current frame in jpg file
        name = os.path.join(frames_folder, "frame") + str(currentFrame) + '.jpg'
        print('Creating...' + str(currentFrame), end="\r")
        cv2.imwrite(name, frame)
        # To stop duplicate images
        currentFrame += 1
    else:
        print("Created {} frames in {}".format(currentFrame - 1, frames_folder))
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
