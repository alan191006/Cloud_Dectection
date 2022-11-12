import csv
import time
import numpy as np
from jtop import jtop
from utils.onnx_helper import ONNXClassifierWrapper

LOGGING_TIME = 30
EXECUTION_TIME = 3600
MODEL_PTH = "./models/model_engine.trt"
CSV_PTH = "./log_trt.csv"

OUTPUT_DIM = [1, 2, 384, 384]
model = ONNXClassifierWrapper("model_engine.trt", OUTPUT_DIM)

def get_stat():

    jetson = jtop()
    jetson.start()

    start_time = time.time()
    last_log = start_time

    with jtop() as jetson:
	
        with open(CSV_PTH, 'w') as csv_file:
            stats = jetson.stats
            writer = csv.DictWriter(csv_file, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)

            while jetson.ok() and (time.time() - start_time < EXECUTION_TIME):

                dummy_input = np.random.rand(1, 4, 384, 384)
                pred = model.predict(dummy_input)
                
                if (time.time() - last_log > LOGGING_TIME):
                    stats = jetson.stats

                    del stats["time"]
                    del stats["uptime"]

                    writer.writerow(stats)
                    print(f"Log at {time.time()}")

                    last_log = time.time()


        csv_file.close()


if __name__ == "__main__":
   
    get_stat()
