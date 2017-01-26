import time
import cv2
import numpy as np
from chainer import serializers, Variable
import chainer.functions as F
import argparse
from yolov2 import *

# argument parse
parser = argparse.ArgumentParser(description="指定したパスの動画ファイルを読み込み、bbox及びクラスの予測を行う")
parser.add_argument('path', help="動画ファイルへのパスを指定")
args = parser.parse_args()

# hyper parameters
input_height, input_width = (416, 416)
#weight_file = "./backup/yolov2_final.model"
weight_file = "./backup/6500.model"
label_file = "../dataset/yolov2_fruits_dataset/label.txt"
video_file = args.path
n_classes = 10
n_boxes = 5
detection_thresh = 0.5
iou_thresh = 0.5

# read labels
with open(label_file, "r") as f:
    labels = f.read().strip().split("\n")

# read video
cap = cv2.VideoCapture(video_file)
codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video_writer = cv2.VideoWriter('yolov2_output_video.avi', codec, 30.0, (input_height, input_width))

# load model
print("loading model...")
model = YOLOv2Predictor(YOLOv2(n_classes=n_classes, n_boxes=n_boxes))
serializers.load_hdf5(weight_file, model) # load saved model
model.predictor.train = False
model.predictor.finetune = False

frame_cnt = 0
while(True):
    ret, orig_img = cap.read()
    orig_img = cv2.resize(orig_img, (input_height, input_width))
    img = np.asarray(orig_img, dtype=np.float32) / 255.0
    img = img.transpose(2, 0, 1)

    # forward
    x_data = img[np.newaxis, :, :, :]
    x = Variable(x_data)
    cuda.get_device(0).use()
    model.to_gpu()
    x.to_gpu()
    x, y, w, h, conf, prob = model.predict(x)

    # parse result
    _, _, _, grid_h, grid_w = x.shape
    x = F.reshape(x, (n_boxes, grid_h, grid_w)).data.get()
    y = F.reshape(y, (n_boxes, grid_h, grid_w)).data.get()
    w = F.reshape(w, (n_boxes, grid_h, grid_w)).data.get()
    h = F.reshape(h, (n_boxes, grid_h, grid_w)).data.get()
    conf = F.reshape(conf, (n_boxes, grid_h, grid_w)).data.get()
    prob = F.transpose(F.reshape(prob, (n_boxes, n_classes, grid_h, grid_w)), (1, 0, 2, 3)).data.get()
    detected_indices = (conf * prob).max(axis=0) > detection_thresh

    results = []
    for i in range(detected_indices.sum()):
        results.append({
            "label": labels[prob.transpose(1, 2, 3, 0)[detected_indices][i].argmax()],
            "probs": prob.transpose(1, 2, 3, 0)[detected_indices][i],
            "conf" : conf[detected_indices][i],
            "objectness": conf[detected_indices][i] * prob.transpose(1, 2, 3, 0)[detected_indices][i].max(),
            "box"  : Box(x[detected_indices][i], y[detected_indices][i], w[detected_indices][i], h[detected_indices][i])
        })

    # nms
    nms_results = []
    for i in range(len(results)):
        overlapped = False
        for j in range(i+1, len(results)):
            if box_iou(results[i]["box"], results[j]["box"]) > iou_thresh:
                overlapped = True
                if results[i]["objectness"] > results[j]["objectness"]:
                    temp = results[i]
                    results[i] = results[j]
                    results[j] = temp
        if not overlapped:
            nms_results.append(results[i])

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
        cv2.putText(orig_img, text, (left, top-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        print(text)

    video_writer.write(orig_img)
    cv2.imwrite("sample_images/%d.jpg" % (frame_cnt), orig_img)
    frame_cnt += 1
    if frame_cnt > 200:
        break

video_writer.release()
