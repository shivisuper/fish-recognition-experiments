import numpy as np
import cv2

rectangle = np.zeros((300, 300), dtype="uint8")
cv2.rectangle(rectangle, (25, 25), (275, 275), 255, -1)
cv2.imshow("Rectangle", rectangle)
cv2.waitKey(0)

circle = np.zeros((300, 300), dtype="uint8")
cv2.circle(circle, (150, 150), 150, 255, -1)
cv2.imshow("Circle", circle)
cv2.waitKey(0)

bitwiseAND = cv2.bitwise_and(rectangle, circle)
cv2.imshow("BITWISE AND", bitwiseAND)
cv2.waitKey(0)

bitwiseOR = cv2.bitwise_or(rectangle, circle)
cv2.imshow("BITWISE OR", bitwiseOR)
cv2.waitKey(0)

bitwiseXOR = cv2.bitwise_xor(rectangle, circle)
cv2.imshow("BITWISE XOR", bitwiseXOR)
cv2.waitKey(0)

bitwiseNOT = cv2.bitwise_not(circle)
cv2.imshow("BITWISE NOT", bitwiseNOT)
cv2.waitKey(0)
