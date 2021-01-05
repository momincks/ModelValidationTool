# Model Validaton Tool
A comprehensive validation pipelines for ONNX/TensorRT models with visualizations, which currently supports:
* comparison between multiple models
* flexibility for models with different input formats and labels
* augmentation for the vaidation dataset
* visualize model outputs for further analysis or presentations
* ONNX accelerations
* ~~TensorRT accelerations~~
***
## Requirements:
* [numpy](https://github.com/numpy/numpy)
* [opencv](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
* [onnxruntime](https://github.com/microsoft/onnxruntime)
* [tensorrt](https://github.com/NVIDIA/TensorRT)
***
## Instructions:
There are 5 JSON configs under ***.config/***:
* **main_config**: inference mode, data augmentation mode ...
* **model_config**: model paths, input formats, label mappings ...
* **dataset_config**: dataset path, dataset, label mappings ...
* **augment_config**: augmentation methods, augmentation parameters ...
* **reporter_config**: output plot size ...
### - dataset preparations:
The validation dataset can be either put under the directory ***./lib/datasets/[custom_name]/*** or located in other place.
As long as the path is correct.
```
"dataset_path": "./lib/datasets/test1/"
data path structure:
test1:
╠ class1
  ╚ *.jpg ...
╠ class2
  ╚ *.jpg ...
╠ class3
  ╚ *.jpg ...
...
```
The class names of class1, class2, class3.. should be defined in ***"label_mapping"*** under ***dataset_config***.
Run ***main.py*** after you prepared configs, datasets and models.
### - model preparations:
The models can be either put under the directory ***./lib/datasets/[custom_name]/*** or located in other place.
As long as the path is correct.
Remember to change the ***model_config*** according to the model's format requirements, which these are currently supported:
* **shape**: NHWC or NCHW
* **channel**: RGB or BGR
* **scale**: 0-1 or 0-255

**Please make sure your label mapping list for that particular model matches its model output in order!**
***
## Augmentations:
**Variety** parameter defines how many augmentation combinations will be generated in this seed.
**It's strongly recommended that this parameter should be > dataset size * dataset_multipier.**
You can save the seed or load a seed by changing:
```
# within main_config.json:
# to save new seed (seed name will be in datetime format):
"load_seed_preset": false
# to load a seed:
"load_seed_preset": "./lib/seeds/[seedname].json"
```
Currently supported data augmentation methods:
*   **horizontal_shift**: intensity, list, min-max range, -1 to 1, +0.2:

![](./demo/horizontal_shift+0.2.jpg)
*   **vertical_shift**: intensity, list, min-max range, -1 to 1, +0.2:

![](./demo/vertical_shift+0.2.jpg)
*   **horizontal_flip**: probablity, integer, 0 to 1:

 ![](./demo/horizontal_flip.jpg)
*   **vertical_flip**: probablity, integer, 0 to 1:

![](./demo/vertical_flip.jpg)
*   **rotation**: intensity, list, min-max range, -1 to 1, +0.2:

![](./demo/rotation+0.2.jpg)
*   **brightness**: intensity, list, min-max range, -1 to 1, +0.2:

![](./demo/brightness+0.2.jpg)
*   **contrast**: intensity, list, min-max range, -1 to 1, +0.2:

![](./demo/contrast+0.2.jpg)
*   **noise_mask**: intensity, list, min-max range, 0 to 1, +0.2:

![](./demo/noise_mask+0.2.jpg)
*   **pixel_attack**: intensity, list, min-max range, 0 to 1, +0.2:

![](./demo/pixel_attack+0.2.jpg)
*   **pixelation**: intensity, list, min-max range, 0 to 1, +0.2:

![](./demo/pixelation+0.2.jpg)

For detailed augmentation effects, please check ***./demo/***. ***Noise_mask*** and ***pixel_attack*** are aggressive methods. Please fine-tune the parameters each time.
***
## Tips:
*   It's highly recommended that to ***turn on data augmentation*** (wild intensity maybe) and set ***"data_multipier" to 3 or even higher***. So the tool will repeat the dataset with different augmentation combinations.
*   Data in datasets should be ***easily labeled by humans***. This way we can test the models with accurate true labels.
