import cv2
import numpy as np
import pandas as pd
import imgaug as ia
import imgaug.augmenters as iaa
import os

IMG_DIR = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\data"
GT_DIR = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\label"
CSV_DIR = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\data.csv"

def augment(IMG_DIR, GT_DIR, CSV_DIR):

    print("Start augmenting!")

    # Read the CSV file as a dataframe
    csv = pd.read_csv(CSV_DIR)

    # Iterate through the existing images and change contrast
    for file in os.listdir(IMG_DIR):
        gt_id = file.replace("img", "gt", 1)
        
        img_dir = os.path.join(IMG_DIR, file)
        gt_dir = os.path.join(GT_DIR, gt_id)
        
        img = cv2.imread(img_dir)
        gt = cv2.imread(gt_dir)
        
        pos_con  = iaa.GammaContrast((1.5, 1.5))
        neg_con = iaa.GammaContrast((0.5, 0.5))
        
        img_pos = pos_con(image=img)
        img_neg = neg_con(image=img)

        img_pos_name = file.replace(".TIF", "_pos_con.TIF")
        gt_pos_name = gt_id.replace(".TIF", "_pos_con.TIF")
        img_neg_name = file.replace(".TIF", "_neg_con.TIF")
        gt_neg_name = gt_id.replace(".TIF", "_neg_con.TIF")

        img_pos_dir = os.path.join(IMG_DIR, img_pos_name)
        gt_pos_dir = os.path.join(GT_DIR, gt_pos_name)
        img_neg_dir = os.path.join(IMG_DIR, img_neg_name)
        gt_neg_dir = os.path.join(GT_DIR, gt_neg_name)
        
        cv2.imwrite(img_pos_dir, img_pos)
        cv2.imwrite(gt_pos_dir, gt)
        cv2.imwrite(img_neg_dir, img_neg)
        cv2.imwrite(gt_neg_dir, gt)

        csv = csv.append({"Data": img_pos_name, "Labels": gt_pos_name}, ignore_index=True)
        csv = csv.append({"Data": img_neg_name, "Labels": gt_neg_name}, ignore_index=True)

    # Iterate through the existing images and vertically flip
    # I use separate for loop to make the number of image grow exponentially
    for file in os.listdir(IMG_DIR):
        gt_id = file.replace("img", "gt", 1)
        
        img_dir = os.path.join(IMG_DIR, file)
        gt_dir = os.path.join(GT_DIR, gt_id)
        
        img = cv2.imread(img_dir)
        gt = cv2.imread(gt_dir)

        v_flip  = iaa.Fliplr()
        
        img_vflip = v_flip(image=img)
        gt_vflip = v_flip(image=gt)

        img_vflip_name = file.replace(".TIF", "_vflip.TIF")
        gt_vflip_name = gt_id.replace(".TIF", "_vflip.TIF")
        
        img_vflip_dir = os.path.join(IMG_DIR, img_vflip_name)
        gt_vflip_dir = os.path.join(GT_DIR, gt_vflip_name)
        
        cv2.imwrite(img_vflip_dir, img_vflip)
        cv2.imwrite(gt_vflip_dir, gt_vflip)

        csv = csv.append({"Data": img_vflip_name, "Labels": gt_vflip_name}, ignore_index=True)

    # Iterate through the existing images and horizontally flip
    for file in os.listdir(IMG_DIR):
        image_id = file
        gt_id = file.replace("img", "gt", 1)
        
        img_dir = os.path.join(IMG_DIR, file)
        gt_dir = os.path.join(GT_DIR, gt_id)
        
        img = cv2.imread(img_dir)
        gt = cv2.imread(gt_dir)
        
        h_flip  = iaa.Flipud()
        
        img_hflip = h_flip(image=img)
        gt_hflip = h_flip(image=gt)

        img_hflip_name = file.replace(".TIF", "_hflip.TIF")
        gt_hflip_name = gt_id.replace(".TIF", "_hflip.TIF")

        img_hflip_dir = os.path.join(IMG_DIR, img_hflip_name)
        gt_hflip_dir = os.path.join(GT_DIR, gt_hflip_name)
        
        cv2.imwrite(img_hflip_dir,img_hflip)
        cv2.imwrite(gt_hflip_dir, gt_hflip)

        csv = csv.append({"Data": img_hflip_name, "Labels": gt_hflip_name}, ignore_index=True)

    # Write the dataframe as a CSV file
    csv.to_csv(CSV_DIR, index=False)

augment(IMG_DIR, GT_DIR, CSV_DIR)
