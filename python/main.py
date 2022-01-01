import Spatial
import filter
import Spectrum
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "test.jpg"

img = cv.imread(IMG_PATH, 0)
img = filter.box_blur(img, (3,3))

cv.imshow('img', img)
cv.waitKey(0)

