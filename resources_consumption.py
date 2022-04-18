import os
import torch
import numpy as np
from torch import nn
from PIL import Image
from pathlib import Path
from torch.nn.utils import prune
from memory_profiler import profile
from torch.utils.data import Dataset, DataLoader

RED_PTH = "/home/alan/Documents/cloud_detection_data/95Cloud/95red/"
IMG = ("/home/alan/Documents/cloud_detection_data/95Cloud/95red/red_patch_45_3_by_5_LC08_L1TP_034047_20160520_20170324_01_T1.TIF", "", "")

RED_PTH_ = Path(RED_PTH)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "/home/alan/Documents/Cloud_Detection/neural network/c_unet_1649683131.7209947.pth"

# Model architect
class depthwiseSeparableConv(nn.Module):

    def __init__(self, nin, nout):
        super(depthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
class C_UNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = self.contract_block(in_channels, 32)
        self.conv2 = self.contract_block(32, 64)
        self.conv3 = self.contract_block(64, 128)

        self.upconv3 = self.expand_block(128, 64, 3, 1)
        self.upconv2 = self.expand_block(64, 32, 3, 1)
        self.upconv1 = self.expand_block(32, out_channels, 3, 1)

        self.out = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid()
        )


    def __call__(self, x):
        # downsampling part
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        upconv3 = self.upconv3(conv3)
        upconv2 = self.upconv2(upconv3)
        upconv1 = self.upconv1(upconv2)

        out = self.out(upconv1)

        return out


    def contract_block(self, in_channels, out_channels):
        contract = nn.Sequential(
            depthwiseSeparableConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        return contract


    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            torch.nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1) 
        )
        return expand

c_unet = C_UNet(4, 2)
model = C_UNet(4, 2)

c_unet.to(DEVICE)
model.to(DEVICE)

c_unet.load_state_dict(torch.load(MODEL_PATH))
model.load_state_dict(torch.load(MODEL_PATH))

param_to_prune = (
    (model.conv1[0].depthwise, "weight"),
    (model.conv1[0].pointwise, "weight"),

    (model.conv2[0].depthwise, "weight"),
    (model.conv2[0].pointwise, "weight"),

    (model.conv3[0].depthwise, "weight"),
    (model.conv3[0].pointwise, "weight"),

    (model.upconv3[0], "weight"),
    (model.upconv3[3], "weight"),

    (model.upconv2[0], "weight"),
    (model.upconv2[3], "weight"),

    (model.upconv1[0], "weight"),
    (model.upconv1[3], "weight"),

    (model.out[0], "weight")
)

prune.global_unstructured(
    param_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.15,
)

class CloudDataset(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, file_list):
        super().__init__()
        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in file_list if not os.path.isdir(f)]
        
    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        files = {
            'red': r_file, 
            'green': r_file.replace('red', 'green'),
            'blue': r_file.replace('red', 'blue'), 
            'nir': r_file.replace('red', 'nir'),
            'gt': r_file.replace('red', 'gt')
        }
        return files
                                       
    def __len__(self):
        return len(self.files)
     
    def open_as_array(self, idx, invert=False, include_nir=False): 
        raw_rgb = np.stack([
            np.array(Image.open(self.files[idx]['red'])),
            np.array(Image.open(self.files[idx]['green'])),
            np.array(Image.open(self.files[idx]['blue'])),
        ], axis=2)
        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)
        if invert:
            raw_rgb = raw_rgb.transpose((2,0,1))
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max) # normalized


    def open_mask(self, idx, add_dims=False):
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask==255, 1, 0)
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask
    
    def __getitem__(self, idx):
        x = torch.tensor(self.open_as_array(idx, invert=True, include_nir=True), dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.long)
        return x, y
    
    def __repr__(self):
        return f'Dataset class with {self.__len__()} files'

test = CloudDataset(RED_PTH_, RED_PTH.replace("red", "green"), RED_PTH.replace("red", "blue"), RED_PTH.replace("red", "nir"), RED_PTH.replace("red", "gt"), IMG)

x, _ = test[0]
x = x[None, :]

@profile
def test():
    a = c_unet(x)
    b = model(x)

if __name__ == "__main__":
    test()