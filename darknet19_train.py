from lib.preprocess import download_image
import time
import cv2
import numpy as np
import chainer
import glob
from chainer import serializers, optimizers, Variable
import chainer.functions as F
from darknet19 import Darknet19

# hyper parameters
input_height, input_width = (224, 224)
train_file = "../dataset/fruits_train_dataset/train.txt"
label_file = "../dataset/fruits_train_dataset/label.txt"
batch_size = 4
max_batches = 10000
learning_rate = 0.1
lr_decay_power = 4
momentum = 0.9
weight_decay = 0.0005

# load dataset
with open(train_file, "r") as f:
    image_files = f.read().strip().split("\n")

with open(label_file, "r") as f:
    labels = f.read().strip().split("\n")

x_train = []
t_train = []
print("loading images")
for image_file in image_files:
    img = cv2.imread(image_file)
    img = cv2.resize(img, (input_height, input_width))
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    x_train.append(img)
    for i, label in enumerate(labels):
        if label in image_file:
            t_train.append(i)
x_train = np.array(x_train)
t_train = np.array(t_train, dtype=np.int32)

# load model
model = Darknet19()
model.train = True

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

# start to train
for batch in range(max_batches):
    batch_mask = np.random.choice(len(x_train), batch_size)
    x = x_train[batch_mask]
    t = t_train[batch_mask]

    y, loss, accuracy = model(x, t)
    print("[batch %d (%d images)] learning rate: %f, loss: %f, accuracy: %f" % (batch+1, (batch+1) * batch_size, optimizer.lr, loss.data, accuracy.data))

    optimizer.zero_grads()
    loss.backward()

    optimizer.lr = learning_rate * (1 - batch / max_batches) ** lr_decay_power # Polynomial decay learning rate
    optimizer.update()
