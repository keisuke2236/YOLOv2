import numpy as np
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
from lib.utils import *
import time

class Darknet19(Chain):

    """
    Darknet19
    - It takes (224, 224, 3) sized image as input
    """

    def __init__(self):
        super(Darknet19, self).__init__(
            ##### common layers for both pretrained and yolo #####
            conv1  = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1),
            bn1    = L.BatchNormalization(32),
            conv2  = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1),
            bn2    = L.BatchNormalization(64),
            conv3  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1),
            bn3    = L.BatchNormalization(128),
            conv4  = L.Convolution2D(128, 64, ksize=1, stride=1, pad=0),
            bn4    = L.BatchNormalization(64),
            conv5  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1),
            bn5    = L.BatchNormalization(128),
            conv6  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1),
            bn6    = L.BatchNormalization(256),
            conv7  = L.Convolution2D(256, 128, ksize=1, stride=1, pad=0),
            bn7    = L.BatchNormalization(128),
            conv8  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1),
            bn8    = L.BatchNormalization(256),
            conv9  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn9    = L.BatchNormalization(512),
            conv10 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0),
            bn10   = L.BatchNormalization(256),
            conv11  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn11    = L.BatchNormalization(512),
            conv12 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0),
            bn12   = L.BatchNormalization(256),
            conv13 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn13   = L.BatchNormalization(512),
            conv14 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),
            bn14   = L.BatchNormalization(1024),
            conv15 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0),
            bn15   = L.BatchNormalization(512),
            conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),
            bn16   = L.BatchNormalization(1024),
            conv17 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0),
            bn17   = L.BatchNormalization(512),
            conv18 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),
            bn18   = L.BatchNormalization(1024),

            ###### new layer
            conv19 = L.Convolution2D(1024, 10, ksize=1, stride=1, pad=0),
        )
        self.train = False

    def __call__(self, x):
        batch_size = x.data.shape[0]

        ##### common layer
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.1)
        h = F.leaky_relu(self.bn4(self.conv4(h)), slope=0.1)
        h = F.leaky_relu(self.bn5(self.conv5(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn6(self.conv6(h)), slope=0.1)
        h = F.leaky_relu(self.bn7(self.conv7(h)), slope=0.1)
        h = F.leaky_relu(self.bn8(self.conv8(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn9(self.conv9(h)), slope=0.1)
        h = F.leaky_relu(self.bn10(self.conv10(h)), slope=0.1)
        h = F.leaky_relu(self.bn11(self.conv11(h)), slope=0.1)
        h = F.leaky_relu(self.bn12(self.conv12(h)), slope=0.1)
        h = F.leaky_relu(self.bn13(self.conv13(h)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bn14(self.conv14(h)), slope=0.1)
        h = F.leaky_relu(self.bn15(self.conv15(h)), slope=0.1)
        h = F.leaky_relu(self.bn16(self.conv16(h)), slope=0.1)
        h = F.leaky_relu(self.bn17(self.conv17(h)), slope=0.1)
        h = F.leaky_relu(self.bn18(self.conv18(h)), slope=0.1)

        ###### new layer
        h = self.conv19(h)
        h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)

        # reshape
        y = F.reshape(h, (batch_size, -1)) 
        return y

class Darknet19Predictor(Chain):
    def __init__(self, predictor):
        super(Darknet19Predictor, self).__init__(predictor=predictor)

    def __call__(self, x, t):
        y = self.predictor(x)

        #y = F.softmax(h)
        #one_hot_t = np.zeros(y.data.shape, dtype=y.dtype.type)
        #one_hot_t[0][t.data[0]] = 1
        #loss = F.mean_squared_error(y, Variable(one_hot_t))
        #print(loss.data)

        loss = F.softmax_cross_entropy(y, t)
        accuracy = F.accuracy(y, t)
        print("----------------------")
        print(np.argmax(y.data, axis=1))
        print(t.data)
        return y, loss, accuracy

    def predict(self, x):
        y = self.predictor(x)
        return F.softmax(y)

