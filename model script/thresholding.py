import cv2 
import pandas as pd
import time
import multiprocessing as mp
import psutil
import numpy as np
from memory_profiler import profile
import cProfile
import re


# Choosing the parameters for the thresholding model
def threshold_param_selection(data, label):
    
    print("Starting threshold selection!")

    # Initialize the IoU_list variable with 0 since max only works if len(list) < 0
    IoU_list = [0]
    best_param = ()

    # Loop through the chosen range of number for each of the parameters
    # Lower threshold
    for lr in range(60, 100):
        for lg in range(60, 100):
            for lb in range(60, 100):
                # High threshold
                for hr in range(60, 100):
                    for hg in range(60, 100):
                        for hb in range(60, 100):           
                            # Initialize variable to measure IoU per batch
                            local_IoU_list = []
                            # Loop through images in batch
                            for file in range(0, len(data) - 1):
                                
                                # Logging
                                print("Lower threshold value(RGB): {:03} {:03} {:03} {:03} {:03} {:03}".format(lr, lg, lb, hr, hg, hb), end="\r")
                                
                                # Read image
                                input_data = cv2.imread(data[file])
                                input_label = cv2.imread(label[file])
                                
                                # Thresholding
                                out = cv2.inRange(input_data, (lr, lg, lb), (hr, hg, hb))

                                # Convert to ndarray to calculate IoU
                                a = (np.array(out) / 255).astype(np.int32)
                                b = (np.array(input_label)[:, :, 0] / 255).astype(np.int32) # Convert 3D grayscale to 2D ndarray

                                # Calculate IoU
                                try:
                                    IoU = np.sum(np.logical_and(a, b)) / np.sum(np.logical_or(a, b))
                                except ZeroDivisionError:
                                    IoU_list.append(0)
                                
                                # Add IoU of this image to the list
                                local_IoU_list.append(IoU)
                            
                            # Ger the average IoU of the whole batch
                            local_average_IoU = np.sum(local_IoU_list) / len(local_IoU_list)
                            
                            # print("Average IoU is {:.10f} for parameters: {:03} {:03} {:03} {:03} {:03} {:03}").format(local_average_IoU, lr, lg, lb, hr, hg, hb)

                            # Assign the IoU if exceed the previous best IoU
                            if local_average_IoU > max(IoU_list):
                                IoU_list.append(local_average_IoU)
                                best_param = (lr, lg, lb, 255, 255, 255)
                

    # Replace NaN with 0
    IoU_list = np.nan_to_num(IoU_list, nan=0)

    # Get the best IoU
    best_IoU = max(iou_list)

    print("Finished threshold selection!")       
    return  best_param


def thresholding(img):
    
    # Assign threshold from the above function
    lower_threshold = (64, 53, 65)
    higher_threshold = (255, 255, 255)

    out = cv2.inRange(img, lower_threshold, higher_threshold)
    return out

# Evaluate the thresholding function
def thresholding_eval(img, gt):

    image = cv2.imread(img)
    ground_truth = cv2.imread(gt)

    out = thresholding(image)

    a = (np.array(out) / 255).astype(np.int32)
    b = (np.array(ground_truth)[:, :, 0] / 255).astype(np.int32)

    try:
        IoU = np.sum(np.logical_and(a, b)) / np.sum(np.logical_or(a, b))
    except ZeroDivisionError:
        IoU = 0

    return IoU

# For testing
if __name__ == '__main__':

    DATA_DIR = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\data"
    LABEL_DIR = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\label"
    CSV_DIR = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\data.csv"

    csv = pd.read_csv(CSV_DIR)
    data = csv.iloc[:, 0]
    label = csv.iloc[:, 1]

    data = DATA_DIR + "\\" + data
    label = LABEL_DIR + "\\" + label

    """# Select the best parameters
    # Note: I don't know how to implement a better method so go through
    #       the dataset of over 60000 image would take over 2 day
    best_iou, best_param = threshold_param_selection()
    print("Best IoU:        ", best_iou)
    print("Best Parameters: ", best_param)
"""
    # Evaluate the IoU
    iou_list = []

    for i in range(0, len(data) - 1):
        iou = thresholding_eval(data[i], label[i])
        iou_list.append(iou)
    
    average_iou = np.sum(iou_list) / len(iou_list)
    print("Average IoU:    ", average_iou) 
    
    # @profile for measuring memory consumption
    @profile
    def measure_resource_consumption():
        random_image = cv2.imread(data[900])
        thresholding(random_image)


    # Measure runtime (not CPU utilization)
    measure_resource_consumption()
    cProfile.run("re.compile('measure_resource_consumption()')")