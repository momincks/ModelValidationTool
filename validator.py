import os
import cv2
import math
import time
import gc
import numpy as np
from tqdm import tqdm
from augmenter import Augmenter
###
if os.path.isfile("./lib/_cache_/onnx.cache"):
    import onnxruntime
    print(f"Currently ONNX-runtime is running on {onnxruntime.get_device()}.")
elif os.path.isfile("./lib/_cache_/trt.cache"):
    raise SystemError("Tensorrt inference is not supported yet.")

class Validator:

    def __init__(self):
        '''validating one single model only. Iteration on all models will be performed on main.py'''

    def _define_attr(self, settings):
        (self.dataset, self.dataset_config, self.model,
        self.model_config, self.augment_config, 
        self.main_config, self.reporter_config, self.seed) = settings
        ###
        if self.main_config["data_augmentation"] and not self.seed:
            raise NameError("Bug in seed generation process.")
        elif self.seed:
            self.augmenter = Augmenter(self.main_config["data_augmentation"], self.seed)
        ###
        self.epsilon = 1e-8
        self.dataset_len = len(self.dataset)
        self.dataset_label = self.dataset_config["label_mapping"]
        self.model_label = self.model_config["label_mapping"]
        self.num_class = len(self.dataset_label)
        self.model_path = self.model_config["model_path"]
        self.input_size = tuple(self.model_config["input_size"])
        self.input_0to1 = (self.model_config["input_divided_by_255"])
        self.input_RGB = (self.model_config["input_RGB_or_BGR"] == "RGB")
        self.input_NCHW = (self.model_config["input_NHWC_or_NCHW"] == "NCHW")
        self.confusion_matrix = np.zeros([self.num_class,self.num_class], dtype=np.int32)

    def _scoring_start(self):
        self.confusion_matrix = np.zeros([self.num_class,self.num_class], dtype=np.int32)
        self.crossentropy = 0
        self.data_process_time = 0.
        self.precision = []
        self.recall = []
        self.predictions = []
        self.true_labels = []
        self.fps = time.time()

    def _scoring_going(self, y_true_integer, y_pred):
        y_pred_integer = np.argmax(y_pred)  ## return y_pred label integer
        y_pred_label = self.model_label[y_pred_integer]
        y_pred_true_integer = self.dataset_label.index(y_pred_label)  ## mapping model outputs to dataset labels.
        self.confusion_matrix[y_true_integer][y_pred_true_integer] += 1
        y_true = self._one_hot(y_true_integer)
        y_pred = np.clip(y_pred, self.epsilon, 1.)
        self.crossentropy -= np.sum(y_true * np.log(y_pred))
        self.predictions.append(y_pred[0])
        self.true_labels.append(y_true)

    def _scoring_end(self):
        for i in range(self.num_class):
            self.precision.append(self.confusion_matrix[i][i]/np.sum(self.confusion_matrix,0)[i])
            self.recall.append(self.confusion_matrix[i][i]/np.sum(self.confusion_matrix,1)[i])
        self.crossentropy = self.crossentropy/self.dataset_len
        self.fps = self.dataset_len/(time.time() - self.fps - self.data_process_time)

    def _one_hot(self, label_integer):
        label = [0.] * self.num_class
        label[label_integer] = 1.
        return np.array(label)

    def _path_split(self, path):
        return path.split("/")[-2]

    def _prepare_onnx_inference(self):
        '''onnx inference for a single onnx model'''
        onnx_session = onnxruntime.InferenceSession(self.model)
        input_name = onnx_session.get_inputs()[0].name
        output_name = onnx_session.get_outputs()[0].name
        return onnx_session, input_name, output_name

    def _image_read(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, self.input_size)
        image = image.astype(np.float32)/255.
        return image
    
    def _image_reformat(self, image):
        image = np.expand_dims(image, 0)
        if not self.input_0to1:
            image = image*255
            image = image.round()
            image = np.clip(image,0,255)
        if self.input_RGB:  ## channel axis must be -1
            image = np.flip(image, -1)
        if self.input_NCHW:
            image = np.transpose(image, [0,3,1,2])  ## NHWC to NCHW
        return image

    def run(self, settings):
        self._define_attr(settings)
        sess, inputs, outputs = self._prepare_onnx_inference()
        self._scoring_start()

        print(f"Testing {self.model_path}:")
        for index, image_path in enumerate(tqdm(self.dataset)):
            data_process_time_start = time.time()

            image = self._image_read(image_path)
            if self.main_config["data_augmentation"]:
                image = self.augmenter.run(image)
                # cv2.imshow("x",image)
                # cv2.waitKey(0)
            image = self._image_reformat(image)
            y_true_integer = self.dataset_label.index(self._path_split(image_path))

            self.data_process_time += time.time()-data_process_time_start
            y_pred = sess.run([outputs], {inputs: image.astype(np.float32)})
            self._scoring_going(y_true_integer, y_pred)

        self._scoring_end()
        #self.augmenter.reset()
        return [self.reporter_config, self.dataset_config, self.model_config, 
                self.predictions, self.true_labels, self.confusion_matrix,
                self.precision, self.recall, self.crossentropy, self.fps]

            




    
        