import argparse
import myutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow("Original", image)

shifted = myutils.translate(image, 25, 89)
cv2.imshow("Shifted down and right", shifted)

shifted = myutils.translate(image, -50, -90)
cv2.imshow("Shifted up and left", shifted)

cv2.waitKey(0)