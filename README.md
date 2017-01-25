# darknet19とYOLOv2（Chainerバージョン）
Joseph Redmonさんの論文はこちら：

[You Only Look Once](https://arxiv.org/abs/1506.02640)

[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)


darknetのオリジナル実装はこちら：

[darknet](http://pjreddie.com/)


## darknet19
オブジェクトのClassification用CNN。VGG16とほぼ同等の精度を維持しつつ、計算量を抑えてVGG16の4倍以上高速に動作する。YOLOv2の特徴抽出器として、事前学習を行う。


### darknet19の特徴
- VGGと同じくほとんどのカーネルを3x3とし、pooling層の後に必ずchannel数を2倍にする。
- GoogLeNetと同じように、3x3の層の間に1x1の層を入れて次元削減を行う。
- 最後のfully-connectの全結合層を取っ払い、代わりにFCN構造を採用。
- Batch Normalizationを全てのConv層に入れる(分散調整用のepsは1e-5、betaを使わない)。
- Conv層ではBiasを使わない。Batch Normの後にBias層を入れる(channel単位でBias共有)。
- softmax関数の後、cross entropyの代わりにsum of squared errorを使う。
  (224x224の学習ではsum of squared errorよりもmean squared errorのほうが収束が早いが、448x448のfinetune時にはsseのほうが効果的)
- 全てのactivation関数について、reluの代わりにleaky reluを使う。(slope=0.1)
- momentum=0.9、weight decay=0.0005を使って学習する。
- learning rateについて、初期値0.1で、4乗のPolynomial decay方式で減衰させる。
- NINと同じようにGlobal Average pooling層を採用し、入力解像度への依存をなくした。


### darknet19の訓練

以下のコマンドでdarknet19のモデルの学習を行う。
変更点は、最終層に`softmax_cross_entropy`を使っている点以外、オリジナルコードと同じ。
`setup.sh`は、必要なディレクトリを作成する初期化スクリプト。
学習が完了すると、`backup/darknet19_final.model`に重みパラメータを保存する。

```
./setup.sh
python darknet19_train.py
```


### darknet19のテスト
以下のコマンドで、学習済みのモデルを使って、指定した画像をカテゴライズする。
テストに使用する学習済み重みのファイル等、ハイパーパラメータを変更する時は、`darknet19_predict.py`を書き換える。
デフォルトでは`backup/darknet19_final.model`の重みパラメータを読み込んでpredictionを行う。

```
python darknet19_predict.py "画像へのパス"
```

### darknet19\_448の訓練
以下のコマンドで、darknet19の学習済みモデル(`backup/backup.model`)を初期値として読み込んで、darknet19_448の訓練を行う。
パスなどは必要に応じてコード内のハイパーパラメータを修正。学習が完了すると、`backup/darknet19_448_final.model`に重みパラメータが保存される。

```
python darknet19_448_train.py
```

### darknet19\_448のテスト
darknet19_predict.py内の重みパラメータファイル及びデータセットのパスを書き換えれば、そのままdarknet19_448のテスト可能。

### darknet19\_448の重み切り出し
`partial_weights.py`のパラメータを修正して、切り出す層及び書き込み先モデルを指定する。
読み込む重みパラメータはデフォルトで`backup/darknet19_448_final.model`になってる。書き出すファイルはデフォルトで`backup/partial.model`になる。
書き出し先のモデル定義も、必要に応じて編集可能。デフォルトではYOLOv2()に重み代入するようにしてる。
(modelとpredictorをYOLOv2GridProbに変更すればそれ用の重み切り出しが可能)
一通り修正した後で、以下のコマンドを実行すればレイヤー切り出しが行われる。

```
python partial_weights.py
```

### yolov2\_grid\_probの訓練
実験用のモデル、各gridの条件付き確率のみpredictionするためのFCN
YOLOv2の全体学習の一部として実験用。
以下のコマンドで訓練可能。ただし、重みパラメータの初期値としてpartial.modelを読み込むので、予め`partial_weights.py`でこれ用の重み切り出しを行っておく必要がある。

```
python yolov2_grid_prob_train.py
```

grid_probの訓練では各gridごとの条件付き確率を求めるので、通常のclassificationより遥かに収束が遅く、デリケート。sum_of_squared_errorを使う場合は誤差値が大きく、学習率を1e-5まで下げないと全く収束しない。原因としては、入力画像の特徴ではなく、gridの場所の位置情報に対応するclassラベルとして学習されてしまう。(例えば右上は必ずapple)
誤差関数にsum_of_squared_errorを使い、かつ学習率を1e-5に固定にするとちょうど良く学習される。
ただし、weight decayを使うと、一定期間学習した後、train lossが逆に増加し続けるので、weight decayは可能ならば使わないべき。


### yolov2\_grid_probのテスト
以下のコマンドで、学習済みのyolov2_grid_probモデルを使って画像の大まかなセグメンテーションを行う。
読み込む重みパラメータファイルはデフォルトで`backup/yolov2_grid_prob_final.model`

```
python yolov2_grid_prob_predict.py "画像ファイルへのパス"
```

predictした結果、`grid_prob_result.jpg`に画像が書き出される。


### yolov2\_bboxの訓練
実験用のモデル。各gridの全てのanchor boxについて、そのconfidenceとbboxの調整値を訓練する。
YOLOv2の全体学習の一部として実験用。
以下のコマンドで訓練可能。ただし、重みパラメータの初期値としてpartial.modelを読み込むので、予め`partial_weights.py`でこれ用の重み切り出しを行っておく必要がある。

```
python yolov2_bbox_train.py
```

オリジナル論文では、iouがthresh以下のpredicted anchor boxに対して、そのconfidenceを0になるよう修正する。その時の係数を論文では1.0にしているが、ここの実装では0.1とする。また、iouがthresh以上のpredicted anchor boxに対しては、そのconfidenceをiouになるように修正するが、論文の係数5.0に対して自分の実装では10.0にしている。

### yolov2\_bbox_probのテスト
以下のコマンドで、学習済みのyolov2_bboxモデルを使って画像のregion proposalを行う。
読み込む重みパラメータファイルはデフォルトで`backup/yolov2_bbox_final.model`

```
python yolov2_bbox_pred.py "画像ファイルへのパス"
```

predictした結果、`bbox_pred_result.jpg`に画像が書き出される。
正解のbboxを緑、best_iouのデフォルトboxを青、predictした結果を赤で描画している。
