# YOLOv2（Chainerバージョン）
Joseph Redmonさんの論文はこちら：

[You Only Look Once](https://arxiv.org/abs/1506.02640)

[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)


darknetのオリジナル実装はこちら：

[darknet](http://pjreddie.com/)


## 訓練済みYOLOv2モデルの実行
<img src="data/prediction_result.jpg">

yolov2学習済みweightsファイルをダウンロードする。

```
wget http://pjreddie.com/media/files/yolo.weights
```

以下のコマンドでweightsファイルをchainer用にパースする。

```
python yolov2_darknet_parser.py yolo.weights
```

以下のコマンドで好きな画像ファイルを指定して物体検出を行う。
検出結果は`yolov2_result.jpg`に保存される。

```
python yolov2_darknet_predict.py data/people.png
```


## YOLOv2でフルーツデータの訓練
<a href="./YOLOv2_fruits_train.md">YOLOv2を使ったフルーツデータセットの訓練手順</a>

## YOLOv2の理論
<a href="./YOLOv2.md">YOLOv2の仕組み解説</a>