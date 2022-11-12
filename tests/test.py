from src.utils.onnx_helper import ONNXClassifierWrapper
import torchmetrics
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os


print("Finished importing...")

MODEL_PATH = "./models/model_engine.trt"
RED_PATH = "./data/95Cloud/95red/"
GREEN_PATH = "./data/95Cloud/95green/"
BLUE_PATH = "./data/95Cloud/95blue/"
NIR_PATH = "./data/95Cloud/95nir/"
GT_PATH = "./data/95Cloud/95gt/"

r_dir = Path(RED_PATH)

#   0%      1 - 30%    30 - 70%    70 - 100%
no_cloud, less_cloud, more_cloud, full_cloud = [], [], [], []

low_threshold = 384*384 * 0.3
high_threshold = 384*384 * 0.7

u = os.listdir(GT_PATH)
# u = u[:int((len(u)/42))]

print("Start cloud density", end="\r")

for file in u[int(len(u)/2):]:

    img = cv2.imread(GT_PATH + file)
    occurrences = np.count_nonzero(img[:, :, 0] == 255)

    if occurrences == 0:
        no_cloud.append(file)
        continue
    elif occurrences < low_threshold:
        less_cloud.append(file)
        continue
    elif occurrences < high_threshold:
        more_cloud.append(file)
        continue
    else:
        full_cloud.append(file)

print("No cloud images:\t",   len(no_cloud))
print("Less cloud images:\t", len(less_cloud))
print("More cloud images:\t", len(more_cloud))
print("Full cloud images:\t", len(full_cloud))
print("Total images:\t\t", int(len(os.listdir(GT_PATH))/2)+1)


def to_path(list):

    for i in range(len(list)):
        list[i] = RED_PATH + list[i].replace("gt", "red")
    return list


no_cloud_list   = to_path(no_cloud)
less_cloud_list = to_path(less_cloud)
more_cloud_list = to_path(more_cloud)
full_cloud_list = to_path(full_cloud)


class CloudDataset():
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, file_list):
        super().__init__()
        self.files = [self.combine_files(
            f, g_dir, b_dir, nir_dir, gt_dir) for f in file_list if not os.path.isdir(f)]

    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        files = {
            'red':   r_file,
            'green': r_file.replace('red', 'green'),
            'blue':  r_file.replace('red', 'blue'),
            'nir':   r_file.replace('red', 'nir'),
            'gt':    r_file.replace('red', 'gt')
        }
        return files

    def __len__(self):
        return len(self.files)

    def open_as_array(self, idx, invert=False):
        raw_rgb = np.stack([
            cv2.imread(self.files[idx]['red'], cv2.IMREAD_GRAYSCALE),
            cv2.imread(self.files[idx]['green'], cv2.IMREAD_GRAYSCALE),
            cv2.imread(self.files[idx]['blue'], cv2.IMREAD_GRAYSCALE),
            cv2.imread(self.files[idx]['nir'], cv2.IMREAD_GRAYSCALE)
        ], axis=2)
        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)  # normalized

    def open_mask(self, idx, add_dims=False):
        raw_mask = cv2.imread(self.files[idx]['gt'], cv2.IMREAD_GRAYSCALE)
        raw_mask = np.where(raw_mask == 255, 1, 0)
        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, idx):
        x = np.expand_dims(np.ascontiguousarray(
            self.open_as_array(idx, invert=True), dtype=np.float32), 0)
        y = np.array(self.open_mask(idx, add_dims=False), dtype=np.float32)
        return x, y

    def __repr__(self):
        return f'Dataset class with {self.__len__()} files'


print("Cloud dataset based on frequency...")

no_cloud_data = CloudDataset(
    RED_PATH, GREEN_PATH, BLUE_PATH, NIR_PATH, GT_PATH, no_cloud_list)
less_cloud_data = CloudDataset(
    RED_PATH, GREEN_PATH, BLUE_PATH, NIR_PATH, GT_PATH, less_cloud_list)
more_cloud_data = CloudDataset(
    RED_PATH, GREEN_PATH, BLUE_PATH, NIR_PATH, GT_PATH, more_cloud_list)
full_cloud_data = CloudDataset(
    RED_PATH, GREEN_PATH, BLUE_PATH, NIR_PATH, GT_PATH, full_cloud_list)

print(no_cloud_data)
print(less_cloud_data)
print(more_cloud_data)
print(full_cloud_data)

# Single input example
x, y = no_cloud_data[0]
print(f'Image: {x.shape}')
print(f'Mask: {y.shape}')

# model

OUT_DIM = [1, 2, 384, 384]
model = ONNXClassifierWrapper(MODEL_PATH, OUT_DIM)


def get_accuracy(dataset, model):
    acc, rec, pre, f1 = [], [], [], []

    accuracy  = torchmetrics.Accuracy()
    recall    = torchmetrics.Recall()
    precision = torchmetrics.Precision()
    f1_score  = torchmetrics.F1Score()

    for i in range(len(dataset)):

        inp, targets = dataset[i]

        predictions = torch.squeeze(torch.from_numpy(
            np.squeeze(model.predict(inp)[:, 1, :, :])))
        targets = torch.from_numpy(targets).int()

        # print(f"{torch.from_numpy(inp).shape}, {predictions.shape}, {targets.shape}", end="\r")

        acc.append(accuracy(predictions, targets))
        rec.append(recall(predictions, targets))
        pre.append(precision(predictions, targets))
        f1.append(f1_score(predictions, targets))

    # return round(np.mean(acc), 4), round(np.mean(rec)), round(np.mean(pre), 4), round(np.mean(f1), 4)
    return np.mean(acc), np.mean(rec), np.mean(pre), np.mean(f1)


no_acc,   no_rec,   no_pre,   no_f1   = get_accuracy(no_cloud_data,   model)
less_acc, less_rec, less_pre, less_f1 = get_accuracy(less_cloud_data, model)
more_acc, more_rec, more_pre, more_f1 = get_accuracy(more_cloud_data, model)
full_acc, full_rec, full_pre, full_f1 = get_accuracy(full_cloud_data, model)

print("\t\t\tAccuracy\tRecall\t\tPrecision\tF1")
print(f"No cloud:\t{no_acc}\t{no_rec}\t{no_pre}\t{no_f1}")
print(f"Less cloud:\t{less_acc}\t{less_rec}\t{less_pre}\t{less_f1}")
print(f"More cloud:\t{more_acc}\t{more_rec}\t{more_pre}\t{more_f1}")
print(f"Full cloud:\t{full_acc}\t{full_rec}\t{full_pre}\t{full_f1}")

thin_cloud, thick_cloud = [], []

thickness_threshold = (0.46875, 0.46875, 0.46875, 0.46875)

for i in range(len(full_cloud_data)):

    x, y = full_cloud_data[i]
    out = cv2.inRange(np.squeeze(x).transpose(1, 2, 0),
                      thickness_threshold, (1, 1, 1, 1))

    try:
        IoU = np.sum(np.logical_and(out, y)) / np.sum(np.logical_or(out, y))
    except ZeroDivisionError:
        thin_cloud.append(full_cloud_list[i])
        continue

    if IoU > 0.5:
        thick_cloud.append(full_cloud_list[i])
    else:
        thin_cloud.append(full_cloud_list[i])

print(len(thin_cloud))
print(len(thick_cloud))

thin_cloud_data = CloudDataset(
    RED_PATH, GREEN_PATH, BLUE_PATH, NIR_PATH, GT_PATH, thin_cloud)
thick_cloud_data = CloudDataset(
    RED_PATH, GREEN_PATH, BLUE_PATH, NIR_PATH, GT_PATH, thick_cloud)

thin_acc, thin_rec, thin_pre, thin_f1 = get_accuracy(thin_cloud_data, model)
thick_acc, thick_rec, thick_pre, thick_f1 = get_accuracy(
    thick_cloud_data, model)

print("\t\tAccuracy\tRecall\t\tPrecision\tF1")
print(f"Thin cloud:\t{thin_acc}\t{thin_rec}\t{thin_pre}\t{thin_f1}")
print(f"Thick cloud:\t{thick_acc}\t{thick_rec}\t{thick_pre}\t{thick_f1}")

# data to plot
acc_list = [no_acc, less_acc, more_acc, full_acc, thin_acc, thick_acc]
rec_list = [no_rec, less_rec, more_rec, full_rec, thin_rec, thick_rec]
pre_list = [no_pre, less_pre, more_pre, full_pre, thin_pre, thick_pre]
f1_list  = [no_f1,  less_f1,  more_f1,  full_f1,  thin_f1,  thick_f1]

name = ["", "No Cloud", "Less Cloud", "More Cloud",
        "Full Cloud", "Thin Cloud", "Thick Cloud"]

s = np.asarray([name,
                ["Accuracy"] + acc_list,
                ["Recall"] + rec_list,
                ["Precision"] + pre_list,
                ["F1"] + f1_list])

np.savetxt("/home/lux/Desktop/Cloud_detection/import.csv",
           s, delimiter=",", fmt='%s')

name_list = ("0% cloud",
             "1 - 30% cloud",
             "30 - 70% cloud",
             "71 - 100% cloud",
             "thin cloud",
             "thick cloud")

n_groups = len(name_list)

# create plot
fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (30, 16)
index = np.arange(n_groups)
bar_width = 0.6

rects1 = plt.bar(index + bar_width*-2, acc_list, bar_width, label="Accuracy")
rects2 = plt.bar(index + bar_width*-1, rec_list, bar_width, label="Recall")
rects3 = plt.bar(index + bar_width*0, pre_list, bar_width, label="Precision")
rects4 = plt.bar(index + bar_width*1, f1_list, bar_width, label="F1")

plt.xlabel("Loss and Accuracy")
plt.ylabel("Value")
plt.title('C-UNet++ performance on various images type')
plt.xticks(index, name_list)
plt.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)
ax.bar_label(rects4, padding=3)

plt.tight_layout()
plt.savefig("/home/alan/Documents/Cloud_Detection/Performance bar chart (hinge loss).png",
            bbox_inches="tight")
plt.show()
