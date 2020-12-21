import cv2
import random
import numpy as np

class Augmenter:

    def __init__(self, enable, seed):
        self.enable = enable
        self.seed = seed
        self.method = [i for i in self.seed if i != "variety"]
        self.variety = self.seed["variety"]
        self.count = dict(zip(self.seed.keys(),[0 for _ in range(len(self.seed.keys()))]))
    
    def horizontal_shift(self, image):
        param = self.seed["horizontal_shift"][self.count["horizontal_shift"]%self.seed["variety"]]
        width = int(image.shape[1]*param)
        if width > 0:  ## shift to right
            image[:,width:,:] = image[:,:-width,:]
        elif width < 0:  ## shift to left
            image[:,:-width,:] = image[:,width:,:]            
        self.count["horizontal_shift"] += 1
        return image

    def vertical_shift(self, image):
        param = self.seed["vertical_shift"][self.count["vertical_shift"]%self.seed["variety"]]
        height = int(image.shape[0]*param)
        if height > 0:  ## downward shift
            image[height:,:,:] = image[:-height,:,:]
        elif height < 0:  ## upward shift
            image[:-height,:,:] = image[height:,:,:]
        self.count["vertical_shift"] += 1
        return image

    def horizontal_flip(self, image):
        param = self.seed["horizontal_flip"][self.count["horizontal_flip"]%self.seed["variety"]]
        if param:
            self.count["horizontal_flip"] += 1
            image = cv2.flip(image, 1)
        return image

    def vertical_flip(self, image):
        param = self.seed["vertical_flip"][self.count["vertical_flip"]%self.seed["variety"]]
        if param:
            self.count["vertical_flip"] += 1
            image = cv2.flip(image, 0)
        return image

    def rotation(self, image):
        param = self.seed["rotation"][self.count["rotation"]%self.seed["variety"]] * 180.
        if param:

            height, width = image.shape[:2]
            cent_x, cent_y = width // 2, height // 2

            mat = cv2.getRotationMatrix2D((cent_x, cent_y), -param, 1.0)
            cos, sin = np.abs(mat[0, 0]), np.abs(mat[0, 1])

            n_width = int((height * sin) + (width * cos))
            n_height = int((height * cos) + (width * sin))

            mat[0, 2] += (n_width / 2) - cent_x
            mat[1, 2] += (n_height / 2) - cent_y

            image = cv2.warpAffine(image, mat, (n_width, n_height))
            new_height, new_width = image.shape[:2]
            image = image[int((new_height-height)/2):int((new_height+height)/2),
                            int((new_width-width)/2):int((new_width+width)/2)]
                            
            self.count["rotation"] += 1
        return image    

    def brightness(self, image): 
        param = self.seed["brightness"][self.count["brightness"]%self.seed["variety"]]        
        image = image + param
        image = np.clip(image, 0., 1.)
        self.count["brightness"] += 1
        return image

    def contrast(self, image):
        param = self.seed["contrast"][self.count["contrast"]%self.seed["variety"]]
        param = (1+param)**(1+param)
        image = np.mean(image) + param * image - param * np.mean(image)
        image = np.clip(image, 0., 1.)
        self.count["contrast"] += 1
        return image

    def noise_mask(self, image):
        param = self.seed["noise_mask"][self.count["noise_mask"]%self.seed["variety"]]
        height, width, channel = image.shape
        noise = np.random.rand(height, width, channel)
        image = image * (1-param) + noise * param
        image = np.clip(image, 0., 1.)
        self.count["noise_mask"] += 1
        return image

    def pixel_attack(self, image):
        param = self.seed["pixel_attack"][self.count["pixel_attack"]%self.seed["variety"]]
        if param:
            height, width, channel = image.shape
            for _ in range(int(height * width * param)):
                image[random.randint(0, height-1), random.randint(0, width-1), :] = np.random.rand(channel)
            self.count["pixel_attack"] += 1
        return image

    def pixelation(self, image):
        param = 1. - self.seed["pixelation"][self.count["pixelation"]%self.seed["variety"]]
        height, width = image.shape[:2]
        image = cv2.resize(image, (int(width*param), int(height*param)), cv2.INTER_NEAREST)
        image = cv2.resize(image, (width, height), cv2.INTER_NEAREST)
        return image

    def reset(self):
        self.count = dict(zip([i for i in self.seed], [0] * len(self.seed)))

    def run(self, image):
        if self.enable:
            for method in self.method:
                image = getattr(self, method)(image)
        return image




        

            
