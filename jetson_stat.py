import csv
import time
import torch
from jtop import jtop

LOGGING_TIME = 30
EXECUTION_TIME = 3600
MODEL_PTH = ""
CSV_PTH = ""

def get_stat():

    model = torch.jit.load(MODEL_PTH)
    jetson = jtop()
    jetson.start()

    start_time = time.time()

    with jtop() as jetson:
        with open(CSV_PTH, "a") as csv_file:
            stats = jetson.stats

            writer = csv.DictWriter(csv, fieldnames=stats.keys)
            writer.writeheader()
            writer.writerow(stats)

            while jetson.ok() and (time.time() - start_time < EXECUTION_TIME):
                stats = jetson.stats

                del stats["time"]
                del stats["uptime"]

                writer.writerow(stats)
                print(f"Log at {stats['time']}")
                
                time.sleep(LOGGING_TIME)

        csv_file.close
            

if __name__ == "__main__":
    
    get_stat()