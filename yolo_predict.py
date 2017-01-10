from lib.preprocess import download_image
import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
from YOLONet import YOLONet

# load model
model = YOLONet()
model.train = False

image_path = download_image()
img = cv2.imread(image_path).astype(np.float32)
img = cv2.resize(img, (448, 448))
x_data = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
x = Variable(x_data)
model(x)
