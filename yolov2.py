import numpy as np
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.links as L
import chainer.functions as F
from lib.utils import *
from lib.functions import *

class YOLOv2(Chain):

    """
    YOLOv2
    - It takes (416, 416, 3) sized image as input
    """

    def __init__(self, n_classes, n_boxes):
        super(YOLOv2, self).__init__(
            ##### common layers for both pretrained layers and yolov2 #####
            conv1  = L.Convolution2D(3, 32, ksize=3, stride=1, pad=1, nobias=True),
            bn1    = L.BatchNormalization(32, use_beta=False),
            bias1  = L.Bias(shape=(32,)),
            conv2  = L.Convolution2D(32, 64, ksize=3, stride=1, pad=1, nobias=True),
            bn2    = L.BatchNormalization(64, use_beta=False),
            bias2  = L.Bias(shape=(64,)),
            conv3  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn3    = L.BatchNormalization(128, use_beta=False),
            bias3  = L.Bias(shape=(128,)),
            conv4  = L.Convolution2D(128, 64, ksize=1, stride=1, pad=0, nobias=True),
            bn4    = L.BatchNormalization(64, use_beta=False),
            bias4  = L.Bias(shape=(64,)),
            conv5  = L.Convolution2D(64, 128, ksize=3, stride=1, pad=1, nobias=True),
            bn5    = L.BatchNormalization(128, use_beta=False),
            bias5  = L.Bias(shape=(128,)),
            conv6  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn6    = L.BatchNormalization(256, use_beta=False),
            bias6  = L.Bias(shape=(256,)),
            conv7  = L.Convolution2D(256, 128, ksize=1, stride=1, pad=0, nobias=True),
            bn7    = L.BatchNormalization(128, use_beta=False),
            bias7  = L.Bias(shape=(128,)),
            conv8  = L.Convolution2D(128, 256, ksize=3, stride=1, pad=1, nobias=True),
            bn8    = L.BatchNormalization(256, use_beta=False),
            bias8  = L.Bias(shape=(256,)),
            conv9  = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn9    = L.BatchNormalization(512, use_beta=False),
            bias9  = L.Bias(shape=(512,)),
            conv10 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn10   = L.BatchNormalization(256, use_beta=False),
            bias10 = L.Bias(shape=(256,)),
            conv11 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn11   = L.BatchNormalization(512, use_beta=False),
            bias11 = L.Bias(shape=(512,)),
            conv12 = L.Convolution2D(512, 256, ksize=1, stride=1, pad=0, nobias=True),
            bn12   = L.BatchNormalization(256, use_beta=False),
            bias12 = L.Bias(shape=(256,)),
            conv13 = L.Convolution2D(256, 512, ksize=3, stride=1, pad=1, nobias=True),
            bn13   = L.BatchNormalization(512, use_beta=False),
            bias13 = L.Bias(shape=(512,)),
            conv14 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn14   = L.BatchNormalization(1024, use_beta=False),
            bias14 = L.Bias(shape=(1024,)),
            conv15 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True),
            bn15   = L.BatchNormalization(512, use_beta=False),
            bias15 = L.Bias(shape=(512,)),
            conv16 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn16   = L.BatchNormalization(1024, use_beta=False),
            bias16 = L.Bias(shape=(1024,)),
            conv17 = L.Convolution2D(1024, 512, ksize=1, stride=1, pad=0, nobias=True),
            bn17   = L.BatchNormalization(512, use_beta=False),
            bias17 = L.Bias(shape=(512,)),
            conv18 = L.Convolution2D(512, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn18   = L.BatchNormalization(1024, use_beta=False),
            bias18 = L.Bias(shape=(1024,)),

            ###### new layer
            conv19 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn19   = L.BatchNormalization(1024, use_beta=False),
            bias19 = L.Bias(shape=(1024,)),
            conv20 = L.Convolution2D(1024, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn20   = L.BatchNormalization(1024, use_beta=False),
            bias20 = L.Bias(shape=(1024,)),
            conv21 = L.Convolution2D(3072, 1024, ksize=3, stride=1, pad=1, nobias=True),
            bn21   = L.BatchNormalization(1024, use_beta=False),
            bias21 = L.Bias(shape=(1024,)),
            #conv22 = L.Convolution2D(1024, n_boxes * (n_classes + 5), ksize=1, stride=1, pad=0),
            conv22 = L.Convolution2D(1024, 10, ksize=1, stride=1, pad=0),
        )
        self.train = False
        self.finetune = False
        self.n_boxes = n_boxes
        self.n_classes = n_classes

    def __call__(self, x):
        ##### common layer
        h = F.leaky_relu(self.bias1(self.bn1(self.conv1(x), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias2(self.bn2(self.conv2(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias3(self.bn3(self.conv3(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias4(self.bn4(self.conv4(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias5(self.bn5(self.conv5(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias6(self.bn6(self.conv6(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias7(self.bn7(self.conv7(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias8(self.bn8(self.conv8(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias9(self.bn9(self.conv9(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias10(self.bn10(self.conv10(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias11(self.bn11(self.conv11(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias12(self.bn12(self.conv12(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias13(self.bn13(self.conv13(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        high_resolution_feature = reorg(h) # 高解像度特徴量をreorgでサイズ落として保存しておく
        h = F.max_pooling_2d(h, ksize=2, stride=2, pad=0)
        h = F.leaky_relu(self.bias14(self.bn14(self.conv14(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias15(self.bn15(self.conv15(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias16(self.bn16(self.conv16(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias17(self.bn17(self.conv17(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias18(self.bn18(self.conv18(h), test=not self.train, finetune=self.finetune)), slope=0.1)

        ###### new layer
        h = F.leaky_relu(self.bias19(self.bn19(self.conv19(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.leaky_relu(self.bias20(self.bn20(self.conv20(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = F.concat((h, high_resolution_feature), axis=1) # output concatnation
        h = F.leaky_relu(self.bias21(self.bn21(self.conv21(h), test=not self.train, finetune=self.finetune)), slope=0.1)
        h = self.conv22(h)

        '''
        # reshape
        batch_size, input_channel, input_height, input_width = h.shape
        h = F.reshape(F.transpose(h, (0, 2, 3, 1)), (batch_size, input_height, input_width, self.n_boxes, -1))
        x, y, w, h, conf, categories = F.split_axis(h, (1, 2, 3, 4, 5), axis=4)

        return x, y, w, h, conf, categories
        '''

        return h
        #h = F.average_pooling_2d(h, h.data.shape[-1], stride=1, pad=0)
        #y = F.reshape(h, (1, -1)) 
        #return y

class YOLOv2Predictor(Chain):
    def __init__(self, predictor):
        super(YOLOv2Predictor, self).__init__(predictor=predictor)
        #self.anchors = np.array([[2.12, 4.04], [3.04, 5.12], [4.08, 2.68], [5.36, 5.53], [3.69, 6.08]])
        self.anchors = np.array([[4.08, 2.68], [3.04, 5.12]])

    def __call__(self, x, t):
        '''
        x, y, w, h, conf, categories = self.predictor(x)
        batch_size, grid_h, grid_w, n_boxes, _ = x.shape  


        # process loss
        categories = F.transpose(F.softmax(F.transpose(categories, (0, 4, 1, 2, 3))), (0, 2, 3, 4, 1)) # softmax(categories)

        t_categories = categories.data.copy()
        for batch in range(batch_size):
            t_categories[batch, :, :, :, :] = 0
            t_categories[batch, :, :, :, t[batch][0]["label"]] = 1

            print(t[batch][0]["one_hot_label"])
            print(categories[batch][0][0][0].data)
            print(categories[batch][0][0][1].data)
            print(categories[batch][10][0][1].data)
        '''


        '''
        #conf = F.sigmoid(conf) # sigmoid(conf)
        #x = F.sigmoid(x)
        #y = F.sigmoid(y)
        #w = F.exp(w)
        #h = F.exp(h)

        #compute best anchor in the grid which truth center is in, then strenthen category
        t_categories = categories.data.copy()
        abs_anchors = self.anchors / np.array([grid_w, grid_h])
        for batch in range(batch_size):
            for object_index in range(len(t[batch])):
                truth_box = t[batch][object_index]
                truth_w = int(float(truth_box["x"]) * grid_w)
                truth_h = int(float(truth_box["y"]) * grid_h)
                truth_n = 0
                best_iou = 0.0
                for anchor_index, abs_anchor in enumerate(abs_anchors):
                    iou = box_iou(Box(0, 0, float(truth_box["w"]), float(truth_box["h"])), Box(0, 0, abs_anchor[0], abs_anchor[1]))
                    if best_iou < iou:
                        best_iou = iou
                        truth_n = anchor_index

                t_categories[batch][truth_h][truth_w][truth_n] = 0
                t_categories[batch][truth_h][truth_w][truth_n][truth_box["label"]] = 1
                print(batch, truth_h, truth_w, truth_n)
                print(categories[batch][truth_h][truth_w][truth_n].data)
                print(truth_box["one_hot_label"])

        t_categories = Variable(t_categories)
        t_categories.to_gpu()

        loss = F.sum(((categories - t_categories) ** 2))
        #loss = sum_of_squared_error(categories, t_categories)
        '''
        

        y = F.softmax(self.predictor(x))
        t_categories = y.data.copy()

        batch_size, n_categories, grid_h, grid_w = y.shape
        for batch in range(batch_size):
            truth_box = t[batch][0]
            truth_w = int(float(truth_box["x"]) * grid_w)
            truth_h = int(float(truth_box["y"]) * grid_h)
            t_categories[batch, :, truth_h, truth_w] = 0
            t_categories[batch, int(truth_box["label"]), truth_h, truth_w] = 1

            print(truth_h, truth_w)
            print(y[batch, :,  truth_h, truth_w].data)
            print(truth_box["one_hot_label"])
            #print(truth_w, truth_h)

        t_categories = Variable(t_categories)
        t_categories.to_gpu()


        #loss = sum_of_squared_error(y, t_categories)
        #loss = F.sum(((categories - t_categories) ** 2))
        loss = F.mean_squared_error(y, t_categories)
        return loss


    def predict(self, x):
        y = self.predictor(x)
        return F.softmax(y)
