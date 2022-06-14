import cv2
import numpy as np
import matplotlib.pyplot as plt
from onnx_helper import ONNXClassifierWrapper

ROOT_IMG = "_patch_27_2_by_6_LC08_L1TP_033036_20170804_20170812_01_T1.TIF"

red = "red" + ROOT_IMG
green = "green" + ROOT_IMG
blue = "blue" + ROOT_IMG
nir = "nir" + ROOT_IMG

img = np.stack([
    cv2.imread(red, cv2.IMREAD_GRAYSCALE),
    cv2.imread(green, cv2.IMREAD_GRAYSCALE),
    cv2.imread(blue, cv2.IMREAD_GRAYSCALE),
    cv2.imread(nir, cv2.IMREAD_GRAYSCALE),
], axis=2)

img = img.transpose(2, 0, 1)
img = np.expand_dims(img, axis=0)
print(img.shape)
img = img / np.iinfo(img.dtype).max

OUT_DIM = [1, 2, 384, 384]
model = ONNXClassifierWrapper("model_engine.trt", OUT_DIM)
img = np.ascontiguousarray(img, dtype=np.float32)

pred = model.predict(img)

def get_mask(pred):
    mask_pred = np.zeros([384,384,3])
    # mask_pred[:,:,0] = pred[:,0,:,:] * 255
    mask_pred[:,:,1] = pred[:,1,:,:] * 255
    return mask_pred

mask = get_mask(pred)

plot = plt.imshow(mask)
plt.show()
