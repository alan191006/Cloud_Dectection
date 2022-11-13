# Cloud_Detection

Cloud segmentation in satellite imagery. 

### Datasets used:
* 38 Cloud, A Cloud Segmentation Dataset: [GitHub](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset) 
* 95 Cloud, An Extension to 38 Cloud Dataset: [GitHub](https://github.com/SorourMo/95-Cloud-An-Extension-to-38-Cloud-Dataset) 

---

## Usage

First, change the working directory to the project: 
 ```bash
 cd [directory/to/project]
 ```

Install the required library. If installation is unsuccessful with PyCUDA, refer to [NVIDIA's documentation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-pycuda) for more information.
 ```bash
 python3 -m pip install -r requirements.txt
 ```

The `./src/to_trt.sh` compile the model to a Jetson-optimized version which will be used for inference.  
Give execute permission: 
 ```bash
chmod +x ./src/to_trt.sh
 ```

Run: 
 ```bash
./src/to_trt.sh
 ```

For inference, execute the code below. The prediction will be store in the `./output` folder with the same name as the original image. 
 ```bash
python3 ./src/predict.py [path/to/image]
 ```