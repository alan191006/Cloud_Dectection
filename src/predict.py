import os
import cv2
import sys
from utils.predict import get_mask, predict

img = cv2.imread(sys.argv[1])
mask = get_mask(predict(img))

filename = sys.argv[1].split("/")[-1]
cv2.imwrite(os.path.join("output", filename), mask)
