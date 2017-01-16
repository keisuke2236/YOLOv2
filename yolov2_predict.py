from lib.preprocess import download_image
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
from yolov2 import *

# load model
model = YOLOv2()
model.train = True

image_path = download_image()
img = cv2.imread(image_path)
img = cv2.resize(img, (416, 416)).astype(np.float32)
x_data = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
x = Variable(x_data)
model(x)
