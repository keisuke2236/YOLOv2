import time
import cv2
import numpy as np
import chainer
import glob
import os
from chainer import serializers, optimizers, Variable, cuda
import chainer.functions as F
from darknet19 import *

# hyper parameters
input_height, input_width = (448, 448)
train_file = "../dataset/fruits_dataset/train.txt"
label_file = "../dataset/fruits_dataset/label.txt"
backup_path = "backup"
batch_size = 16
max_batches = 3000
learning_rate = 0.001
lr_decay_power = 4
momentum = 0.9
weight_decay = 0.0005

# load dataset
with open(train_file, "r") as f:
    image_files = f.read().strip().split("\n")

with open(label_file, "r") as f:
    labels = f.read().strip().split("\n")


x_train = []
t_train = [] # normal label
t_train_one_hot = [] # one hot label
print("loading image datasets...")
for image_file in image_files:
    img = cv2.imread(image_file)
    img = cv2.resize(img, (input_height, input_width))
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    x_train.append(img)
    t_train_one_hot.append(np.zeros(len(labels)))
    for i, label in enumerate(labels):
        if label in image_file:
            t_train.append(i)
            t_train_one_hot[-1][i] = 1
x_train = np.array(x_train)
t_train = np.array(t_train, dtype=np.int32)
t_train_one_hot = np.array(t_train_one_hot, dtype=np.float32)

# load model
print("loading model...")
model = Darknet19Predictor(Darknet19())
backup_file = "%s/backup.model" % (backup_path)
if os.path.isfile(backup_file):
    serializers.load_hdf5(backup_file, model) # load saved model
model.predictor.train = True
model.predictor.finetune = True

if hasattr(cuda, "cupy"):
    cuda.get_device(0).use()
    model.to_gpu() # for gpu

optimizer = optimizers.MomentumSGD(lr=learning_rate, momentum=momentum)
optimizer.use_cleargrads()
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))


# start to train
print("start training")
for batch in range(max_batches):
    batch_mask = np.random.choice(len(x_train), batch_size)
    x = Variable(x_train[batch_mask])
    #t = Variable(t_train[batch_mask]) # use normal label and softmax_cross_entropy
    t = Variable(t_train_one_hot[batch_mask]) # use one hot label and squared error
    if hasattr(cuda, "cupy"):
        x.to_gpu() # for gpu
        t.to_gpu() # for gpu

    y, loss, accuracy = model(x, t)
    print("[batch %d (%d images)] learning rate: %f, loss: %f, accuracy: %f" % (batch+1, (batch+1) * batch_size, optimizer.lr, loss.data, accuracy.data))

    optimizer.zero_grads()
    loss.backward()

    optimizer.lr = learning_rate * (1 - batch / max_batches) ** lr_decay_power # Polynomial decay learning rate
    optimizer.update()

    # save model
    if (batch+1) % 500 == 0:
        model_file = "%s/%s.model" % (backup_path, batch+1)
        print("saving model to %s" % (model_file))
        serializers.save_hdf5(model_file, model)
        serializers.save_hdf5(backup_file, model)

print("saving model to %s/final.model" % (backup_path))
serializers.save_hdf5("%s/final.model" % (backup_path), model)
