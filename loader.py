import os
import json
import random
from glob import glob
from datetime import datetime

class Loader:
    """load everything into RAM for later process."""

    def __init__(self, onnx_or_trt="onnx"):
        self.onnx_or_trt = onnx_or_trt
        ###
        self.model_config_path = "./config/model_config.json"
        self.dataset_config_path = "./config/dataset_config.json"
        self.augment_config_path = "./config/augment_config.json"
        self.main_config_path = "./config/main_config.json"
        self.reporter_config_path = "./config/reporter_config.json"
        ###
        self.model_config = self._read_json(self.model_config_path)
        self.dataset_config = self._read_json(self.dataset_config_path)
        self.augment_config = self._read_json(self.augment_config_path)
        self.main_config = self._read_json(self.main_config_path)
        self.reporter_config = self._read_json(self.reporter_config_path)
        ###
        self.model_path = [i.get("model_path") for i in self.model_config]
        self.dataset_path = os.path.join(self.dataset_config["dataset_path"], "*/*.*")
        self.dataset_path = glob(self.dataset_path) * self.dataset_config["dataset_multiplier"]
        self.seed_path = self.main_config["load_seed_preset"]
        ###
        self.number_of_models = len(self.model_path)
        self.min_is_zero_augment = ["horizontal_flip","vertical_flip","gaussian_blur",
                                    "noise_mask","pixel_attack","pixelation"]

    def _read_json(self, path):
        ## read config.json
        config = open(path, "rb").read()
        config = json.loads(config)
        return config

    def generate_seed(self):
        ## to have same params for different models during data augmentation
        using_method, using_seed = [], []  ## not all augmentation methods will be used
        for method in self.augment_config:
                param = self.augment_config.get(method)
                if method == "variety" and isinstance(param,int):
                    using_method.append(method)
                    using_seed.append(param)
                elif isinstance(param,list):
                    if param[1] - param[0] >= 0:
                        using_method.append(method)
                        using_seed.append([random.uniform(param[0],param[1]) for _ in range(self.augment_config["variety"])])
                    else:
                        ValueError(f"wrong range for {method}, {param[0]} to {param[1]}")
                elif isinstance(param,(int,float)):
                    if param:
                        using_method.append(method)
                        using_seed.append([random.random() < param for _ in range(self.augment_config["variety"])])
        seed = dict(zip(using_method,using_seed))
        #count = dict(zip(using_method,[0]*len(using_method)))
        ###
        seed_file_name = str(datetime.now().replace(microsecond=0)).replace(" ","_")
        try:
            f = open(f"./lib/seeds/{seed_file_name}.json", "w", encoding="utf-8")
            json.dump(seed, f, ensure_ascii=False, indent=4)
            f.close()
        except:
            raise EnvironmentError("writing seed files failed.")
        self.seed = seed
        print(f"New seed {seed_file_name}.json is generated.")

    def check(self):
        
        print("Loading user settings...")
        ## check if config exists
        if not all(os.path.exists(i) for i in [self.model_config_path, self.dataset_config_path,
                                                self.augment_config_path, self.main_config_path,
                                                self.reporter_config_path]):
            raise LookupError("Some configs are missing. They should be located in config directory.")        

        ## check if model exists
        if not all(os.path.exists(i) for i in self.model_path):
            raise LookupError("Wrong model paths. Models cannot be found.")

        ## check if dataset exists
        if not len(glob(f"./lib/models/{self.onnx_or_trt}/*.{self.onnx_or_trt}")):
            raise LookupError("Nothing in the dataset. Wrong path or no data?")

        ## check if seed exists
        if not os.path.exists(self.seed_path):
            raise LookupError(("Wrong seed path. Please correct the seed path in main config. " 
                                "You can change \"load_seed_preset\" to False if you want a new one."))
        elif self.main_config["data_augmentation"] and self.seed_path:
            print(f"Loading seed from {self.seed_path}...")
            self.seed = self._read_json(self.seed_path)
        elif not self.main_config["data_augmentation"]:
            print("Data augmentation is turned off. No seed is loaded.")

        ## check if augmentation config is in the same format
        if not self.main_config["load_seed_preset"] and self.main_config["data_augmentation"]:
            for method in self.augment_config:
                param = self.augment_config.get(method)
                if isinstance(param,list):
                    if method in self.min_is_zero_augment:
                        if any([i > 1. or i < 0. for i in param]):
                            raise ValueError(f"{method} parameters should be [min,max] and within range [0,1]")
                    else:
                        if any([i > 1. or i < -1. for i in param]):
                            raise ValueError(f"{method} parameters should be [min,max] and within range [-1,1]")
                if isinstance(param,(int,float)):
                    if method in self.min_is_zero_augment:
                        if param > 1. or param < 0.:
                            raise ValueError(f"{method} parameters should be int/float and within range [0,1]")
            print("No seed preset is defined. Generating seed...")
            self.generate_seed()

            if self.seed["variety"] < len(self.dataset_path):
                print("Seed variety < numbers of data * dataset multiplier. "
                        "Consider to generate a biggest seed with larger seed variety.")

        if len(self.dataset_path) < 2000:
            print("Dataset is too small (<2000). FPS may not be accurate.")

        ## define inference mode
        mode = self.main_config["onnx_or_trt"]
        os.makedirs("./lib/_cache_/", exist_ok=True)
        open(f"./lib/_cache_/{mode}.cache","w").close()

    def run(self, model_index):
        if not hasattr(self, "seed"):
            self.seed = []
        return [
                    self.dataset_path, 
                    self.dataset_config, 
                    self.model_path[model_index], 
                    self.model_config[model_index],
                    self.augment_config,
                    self.main_config,
                    self.reporter_config,
                    self.seed
                ]


        

        


        
        

        

        


