import numpy as np
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
from lib.utils import *
import time

total_cost = 0

class YOLONet(Chain):

    """
    YOLONet
    - It takes (448, 448, 3) sized image as input
    """

    def __init__(self):
        super(YOLONet, self).__init__(
            ##### pretrained layer #####
            conv1  = L.Convolution2D(3, 64, ksize=7, stride=2, pad=3),
            bn1    = L.BatchNormalization(64),
            conv2  = L.Convolution2D(64, 192, ksize=3, stride=1, pad=1),
            bn2    = L.BatchNormalization(192),
            conv3  = L.Convolution2D(192, 128, ksize=1, stride=1, pad=0),
            bn3    = L.BatchNormalization(128),
            conv4  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1),
            bn4    = L.BatchNormalization(256),
            conv5  = L.Convolution2D(256, 256, ksize=1, stride=1, pad=0),
            bn5    = L.BatchNormalization(256),
            conv6  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn6    = L.BatchNormalization(512),
            conv7  = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0),
            bn7    = L.BatchNormalization(256),
            conv8  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn8    = L.BatchNormalization(512),
            conv9  = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0),
            bn9    = L.BatchNormalization(256),
            conv10 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn10   = L.BatchNormalization(512),
            conv11  = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0),
            bn11    = L.BatchNormalization(256),
            conv12 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn12   = L.BatchNormalization(512),
            conv13  = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0),
            bn13    = L.BatchNormalization(256),
            conv14 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1),
            bn14   = L.BatchNormalization(512),
            conv15 = L.Convolution2D(512, 512, ksize=1, stride=1, pad=0),
            bn15   = L.BatchNormalization(512),
            conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),
            bn16   = L.BatchNormalization(1024),
            conv17 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0),
            bn17   = L.BatchNormalization(512),
            conv18 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),
            bn18   = L.BatchNormalization(1024),
            conv19 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0),
            bn19   = L.BatchNormalization(512),
            conv20 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1),
            bn20   = L.BatchNormalization(1024),

            ##### new layer #####
            conv21 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1),
            bn21   = L.BatchNormalization(1024),
            conv22 = L.Convolution2D(1024, 1024, ksize=3, stride=2, pad=1),
            bn22   = L.BatchNormalization(1024),
            conv23 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1),
            bn23   = L.BatchNormalization(1024),
            conv24 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1),
            bn24   = L.BatchNormalization(1024),
        )
        self.train = False

    def __call__(self, x):
        global total_cost

        print('--------- YOLOv1 ----------------------------------------------------------------')
        start_time = time.time()
        shape_before = x.shape
        h = F.leaky_relu(self.bn1(self.conv1(x)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv1', self.conv1, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        shape_after = h.shape
        total_cost += print_pooling_info('pooling', 2, 2, 0, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn2(self.conv2(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv2', self.conv2, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        shape_after = h.shape
        total_cost += print_pooling_info('pooling', 2, 2, 0, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn3(self.conv3(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv3', self.conv3, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn4(self.conv4(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv4', self.conv4, shape_before, shape_after, time.time() - start_time)
        
        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn5(self.conv5(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv5', self.conv5, shape_before, shape_after, time.time() - start_time)
        
        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn6(self.conv6(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv6', self.conv6, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        shape_after = h.shape
        total_cost += print_pooling_info('pooling', 2, 2, 0, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn7(self.conv7(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv7', self.conv7, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn8(self.conv8(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv8', self.conv8, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn9(self.conv9(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv9', self.conv9, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn10(self.conv10(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv10', self.conv10, shape_before, shape_after, time.time() - start_time)
        
        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn11(self.conv11(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv11', self.conv11, shape_before, shape_after, time.time() - start_time)
        
        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn12(self.conv12(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv12', self.conv12, shape_before, shape_after, time.time() - start_time)
        
        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn13(self.conv13(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv13', self.conv13, shape_before, shape_after, time.time() - start_time)
        
        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn14(self.conv14(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv14', self.conv14, shape_before, shape_after, time.time() - start_time)
        
        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn15(self.conv15(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv15', self.conv15, shape_before, shape_after, time.time() - start_time)
        
        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn16(self.conv16(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv16', self.conv16, shape_before, shape_after, time.time() - start_time)
        
        start_time = time.time()
        shape_before = h.shape
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        shape_after = h.shape
        total_cost += print_pooling_info('pooling', 2, 2, 0, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn17(self.conv17(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv17', self.conv17, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn18(self.conv18(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv18', self.conv18, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn19(self.conv19(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv19', self.conv19, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn20(self.conv20(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv20', self.conv20, shape_before, shape_after, time.time() - start_time)

        #####

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn21(self.conv21(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv21', self.conv21, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn22(self.conv22(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv22', self.conv22, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn23(self.conv23(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv23', self.conv23, shape_before, shape_after, time.time() - start_time)

        start_time = time.time()
        shape_before = h.shape
        h = F.leaky_relu(self.bn24(self.conv24(h)), slope=0.1)
        shape_after = h.shape
        total_cost += print_cnn_info('conv24', self.conv24, shape_before, shape_after, time.time() - start_time)

        return h
