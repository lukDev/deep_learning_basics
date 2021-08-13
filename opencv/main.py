import cv2 as cv
import numpy as np

img = np.zeros((512, 512, 3), np.uint8)
cv.line(img, (100, 100), (200, 200), (0, 255, 0), 10)

cv.imshow("Display window", img)
cv.waitKey()
