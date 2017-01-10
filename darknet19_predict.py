from lib.preprocess import download_image
import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
from darknet19 import Darknet19

# load model
model = Darknet19()
model.train = False

image_path = download_image()
img = cv2.imread(image_path).astype(np.float32)
img = cv2.resize(img, (224, 224))
x_data = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
x = Variable(x_data)
model(x)
