import argparse
import cv2
import myutils

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow("Original", image)

rotated = myutils.rotate(image, 45)
cv2.imshow("Rotated by 45 degrees", rotated)

rotated = myutils.rotate(image, -90)
cv2.imshow("Rotated by -90 degrees", rotated)

rotated = myutils.rotate(image, 180)
cv2.imshow("Rotated by 180 degrees", rotated)

cv2.waitKey(0)