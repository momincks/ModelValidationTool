import cv2
import random
import numpy as np
from datetime import datetime

class Augmenter:

    def horizontal_shift(image, param):
        width = int(image.shape[1]*param)
        if width > 0:  ## shift to right
            image[:,width:,:] = image[:,:-width,:]
        elif width < 0:  ## shift to left
            image[:,:-width,:] = image[:,width:,:]            
        return image

    def vertical_shift(image, param):
        height = int(image.shape[0]*param)
        if height > 0:  ## downward shift
            image[height:,:,:] = image[:-height,:,:]
        elif height < 0:  ## upward shift
            image[:-height,:,:] = image[height:,:,:]
        return image

    def horizontal_flip(image, param):
        if param:
            image = cv2.flip(image, 1)
        return image

    def vertical_flip(image, param):
        if param:
            image = cv2.flip(image, 0)
        return image

    def rotation(image, param):
        param *= 180
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
        image = image[int((new_height-height)/2):int((new_height+height)/2),int((new_width-width)/2):int((new_width+width)/2)]

        return image    

    def brightness(image, param): 
        image = image + param
        image = np.clip(image, 0., 1.)
        return image

    def contrast(image, param):
        param = (1+param)**(1+param)
        image = np.mean(image) + param * image - param * np.mean(image)
        image = np.clip(image, 0., 1.)
        return image

    def noise_mask(image, param):
        height, width, channel = image.shape
        noise = np.random.rand(height, width, channel)
        image = image * (1-param) + noise * param
        image = np.clip(image, 0., 1.)
        return image

    def pixel_attack(image, param):
        if param:
            height, width, channel = image.shape
            for num in range(int(height*width*param)):
                image[random.randint(0, height-1), random.randint(0, width-1), :] = np.random.rand(channel)
        return image

    def pixelation(image, param):
        param = 1-param
        height, width = image.shape[:2]
        image = cv2.resize(image, (int(width*param), int(height*param)), cv2.INTER_NEAREST)
        image = cv2.resize(image, (width, height), cv2.INTER_NEAREST)
        return image

    def low_pass_filter(image, param):
        height, width = image.shape[:2]
        longest_side = max(height, width)
        param = longest_side * (1-param)
        cent_y, cent_x = height//2, width//2
        signal = np.fft.fft2(image, axes=(0,1))
        for y in range(height):
            for x in range(width):
                if ((x-cent_x)**2+(y-cent_y)**2)**0.5 < param:
                    signal[y,x,:] *= np.random.rand(3)
        image = np.fft.ifft2(signal, axes=(0,1))
        image = np.abs(image)
        return image

all_functions = [
                "horizontal_shift","vertical_shift","horizontal_flip","vertical_flip",
                "rotation","brightness","contrast","noise_mask",
                "pixel_attack","pixelation"
                ]

all_params = [0.2,0.2,1,1,0.2,0.2,0.2,0.2,0.2,0.2]

# all_functions = ["low_pass_filter"]
# all_params = [0.2]

#img = cv2.imread("lib/datasets/test1/dumptruck_closed/swh_m01_dumptruck_closed_cam1-48_b0_2d_07_6b_e4-2020-10-19-08_43_27.129808+08_00__378-165-930-431.jpg")/255.
img = cv2.imread("lib/husky.jpg")/255.
img = cv2.resize(img,(224,224))
for i, func in enumerate(all_functions):
    aug = img.copy()
    aug = getattr(Augmenter, func)(aug, all_params[i])
    aug = aug*255
    aug.round().astype(np.uint8)
    cv2.imwrite(f"./demo/{func}+{all_params[i]}.jpg",aug)

