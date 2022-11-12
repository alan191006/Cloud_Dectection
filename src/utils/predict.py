import numpy as np
from onnx_helper import ONNXClassifierWrapper

def predict(img):
    
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    out_dim = [1, 2, img.shape[2], img.shape[3]]
    img = img / np.iinfo(img.dtype).max

    model = ONNXClassifierWrapper("./models/model_engine.trt", out_dim)
    img = np.ascontiguousarray(img, dtype=np.float32)

    pred = model.predict(img)
    return pred

def get_mask(pred):
    mask_pred = np.zeros([384,384,3])
    mask_pred[:,:,1] = pred[:,1,:,:] * 255
    return mask_pred