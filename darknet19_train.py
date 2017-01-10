from lib.preprocess import download_image
import time
import cv2
import numpy as np
import chainer
from chainer import serializers, optimizers, Variable
import chainer.functions as F
from darknet19 import Darknet19

# download image
image_path = download_image()
img = cv2.imread(image_path).astype(np.float32)
img = cv2.resize(img, (224, 224))
x_data = img.transpose(2, 0, 1)[np.newaxis, :, :, :]
x = Variable(x_data)

t_data = np.array([9], dtype=np.int32)
t = Variable(t_data)

# hyper parameter
max_batches = 10000
learning_rate = 0.1
lr_decay_power = 4
momentum = 0.9
weight_decay = 0.0005

# load model
model = Darknet19()
model.train = True

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))

# start to learn
for batch in range(max_batches):
    y, loss, accuracy = model(x, t)

    optimizer.zero_grads()
    loss.backward()
    print(loss.data)

    optimizer.lr = learning_rate * (1 - batch / max_batches) ** lr_decay_power # Polynomial decay learning rate
    optimizer.update()
