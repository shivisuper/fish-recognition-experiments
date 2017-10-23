import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

mask = np.zeros(image.shape[:2], dtype="uint8")
(cX, cY) = (image.shape[1] // 2, image.shape[0] // 2)
cv2.rectangle(mask, (cX - 75, cY - 75), (cX + 75, cY + 75), 255, -1)
masked = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Mask applied to the image", masked)

circular_mask = np.zeros(image.shape[:2], dtype="uint8")
cv2.circle(circular_mask, (cX, cY), 100, 255, -1)
masked = cv2.bitwise_and(image, image, mask=circular_mask)
cv2.imshow("Circular mask", masked)
cv2.waitKey(0)
