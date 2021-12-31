import Spatial
import Spectrum
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

IMG_PATH = "test.jpg"

img = cv.imread(IMG_PATH, 0)
img = Spatial.resize(img, (1000, 1000))

cv.imshow('img', img)
cv.waitKey(0)

