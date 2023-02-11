import os
import cv2
import sys
import argparse

from utils.predict import get_mask, predict

parser = argparse.ArgumentParser(description="Prediction script")

parser.add_argument("-m", action="store", dest="model", default=0)
parser.add_argument("-i", action="store", dest="img", default=0)
args = parser.parse_args()

assert os.path.exists(args.model)
assert os.path.exists(args.img)

img = cv2.imread(sys.argv[1])
mask = get_mask(predict(args.model, img))

filename = args.img.split("/")[-1]
cv2.imwrite(os.path.join("output", filename), mask)
