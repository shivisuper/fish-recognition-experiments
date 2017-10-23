from matplotlib import pyplot as plt
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray", image)

hist = cv2.calcHist([image], [0], None, [256], [0, 256])
fig = plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Bins")
plt.ylabel("# of pixels")
plt.plot(hist)
plt.xlim([0, 256])
fig.show()
# plt.show(fig)
cv2.waitKey(0)
cv2.destroyAllWindows()
