import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

flipped = cv2.flip(image, 1)
cv2.imshow("Flipped horizontally", flipped)

flipped = cv2.flip(image, 0)
cv2.imshow("Flipped vertically", flipped)

flipped = cv2.flip(image, -1)
cv2.imshow("Flipped vertically and horizontally", flipped)
cv2.waitKey(0)
