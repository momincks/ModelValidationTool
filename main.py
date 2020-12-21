import os
import sys
sys.tracebacklimit = 0
import warnings
from glob import glob
from loader import Loader
from validator import Validator
from reporter import Reporter

def clear_cache():
    for i in glob("./lib/_cache_/*.cache"):
        os.remove(i)

if __name__ == "__main__":
    print("---Model Validation Tool---")
    clear_cache()
    loader = Loader(onnx_or_trt="onnx")
    loader.check()
    validator = Validator()
    reporter = Reporter()
    for index, model in enumerate(range(loader.number_of_models)):
        settings = loader.run(index)
        metrics = validator.run(settings)
        reporter.run(metrics)
