import os
import shutil
import numpy as np

# Use a csv created by the dataset provider to remove image has less than 60% color pixel
def remove_blank(IMG_DIR, DES_IMG_DIR, CSV_DIR, NAME):

    print("Start removing blank!")

    # Check input parameters
    if not os.path.exists(IMG_DIR):
        print("Please check image directory. Function failed.\n")
        return

    if not os.path.exists(CSV_DIR):
        print("Please check csv file's directory. Function failed.\n")
        return

    if os.path.exists(DES_IMG_DIR):
        dst = DES_IMG_DIR
    else: os.mkdir(DES_IMG_DIR)

    # Read the csv file to an ndarray
    file_to_keep = np.genfromtxt(CSV_DIR, dtype=str)
    
    # Remove the first value: "name"
    file_to_keep = np.delete(file_to_keep, 0)

    # Iterate through image files in the directory 
    for file in os.listdir(IMG_DIR):
        # Iterate through the arrays and compare to the image folder to select which image to keep
        for row in file_to_keep:
            image_id = NAME + "_" + row + ".TIF"
        
            if file == image_id:
                src = os.path.join(IMG_DIR, image_id)
                dst = os.path.join(DES_IMG_DIR, image_id)
                
                # Copy the image matching the given id to the chosen directory
                shutil.copy2(src, dst)


red = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\95red\\"
green = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\95green\\"
blue = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\95blue\\"
nir = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\95nir\\"
gt = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\95gt\\"

train_95_csv = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\38-cloud_95-cloud_training_informative_patches\training_patches_95-cloud_nonempty.csv"

dst_red = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\95red_filtered\\"
dst_green = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\95green_filtered\\"
dst_blue = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\95blue_filtered\\"
dst_nir = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\95nir_filtered\\"
dst_gt = r"C:\Users\hdmqu\Documents\LUX Aerobot\Dataset\Full dataset\95gt_filtered\\"

remove_blank(red, dst_red, train_95_csv, "red")
remove_blank(green, dst_green, train_95_csv, "green")
remove_blank(blue, dst_blue, train_95_csv, "blue")
remove_blank(nir, dst_nir, train_95_csv, "nir")
remove_blank(gt, dst_gt, train_95_csv, "gt")