import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from yolov2 import *

# argument parse
parser = argparse.ArgumentParser(description="指定したパスの画像を読み込み、bbox及びクラスの予測を行う")
parser.add_argument('path', help="画像ファイルへのパスを指定")
args = parser.parse_args()

# hyper parameters
#input_height, input_width = (416, 416)
weight_file = "./yolov2_darknet.model"
image_file = args.path
n_classes = 80
n_boxes = 5
detection_thresh = 0.5
iou_thresh = 0.5
anchors = [[0.738768, 0.874946], [2.42204, 2.65704], [4.30971, 7.04493], [10.246, 4.59428], [12.6868, 11.8741]]
labels = ["person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"]

# read image
print("loading image...")
orig_img = reshape_to_yolo_size(cv2.imread(image_file))
input_height, input_width, _ = orig_img.shape
img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
img = np.asarray(orig_img, dtype=np.float32) / 255.0
img = img.transpose(2, 0, 1)

# load model
print("loading model...")
yolov2 = YOLOv2(n_classes=n_classes, n_boxes=n_boxes)
serializers.load_hdf5(weight_file, yolov2) # load saved model
model = YOLOv2Predictor(yolov2)
model.init_anchor(anchors)
model.predictor.train = False
model.predictor.finetune = False

# forward
x_data = img[np.newaxis, :, :, :]
x = Variable(x_data)
x, y, w, h, conf, prob = model.predict(x)

# parse results
_, _, _, grid_h, grid_w = x.shape
x = F.reshape(x, (n_boxes, grid_h, grid_w)).data
y = F.reshape(y, (n_boxes, grid_h, grid_w)).data
w = F.reshape(w, (n_boxes, grid_h, grid_w)).data
h = F.reshape(h, (n_boxes, grid_h, grid_w)).data
conf = F.reshape(conf, (n_boxes, grid_h, grid_w)).data
prob = F.transpose(F.reshape(prob, (n_boxes, n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data
detected_indices = (conf * prob).max(axis=0) > detection_thresh

results = []
for i in range(detected_indices.sum()):
    results.append({
        "label": labels[prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax()],
        "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
        "conf" : conf[detected_indices][i],
        "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
        "box"  : Box(x[detected_indices][i], y[detected_indices][i], w[detected_indices][i], h[detected_indices][i]).crop_region(input_height, input_width)
    })

# nms
nms_results = nms(results, iou_thresh)

# draw result
for result in nms_results:
    left, top = result["box"].int_left_top()
    cv2.rectangle(
        orig_img,
        result["box"].int_left_top(), result["box"].int_right_bottom(),
        (0, 0, 255),
        2
    )
    text = '%s(%2d%%)' % (result["label"], result["probs"].max()*result["conf"]*100)
    cv2.putText(orig_img, text, (left, top-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    print(text)

print("save results to yolov2_result.jpg")
cv2.imwrite("yolov2_result.jpg", orig_img)
cv2.imshow("w", orig_img)
cv2.waitKey()
