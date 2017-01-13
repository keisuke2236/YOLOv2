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
- Batch Normalizationを全てのConv層に入れる。
- Conv層ではBiasを使わない。Batch Normの後にBias層を入れる(channel単位でBias共有)。
- softmax関数の後、cross entropyの代わりにsum of squared errorを使う。
- 全てのactivation関数について、reluの代わりにleaky reluを使う。(slope=0.1)
- momentum=0.9、weight decay=0.0005を使って学習する。
- learning rateについて、初期値0.1で、4乗のPolynomial decay方式で減衰させる。
- NINと同じようにGlobal Average pooling層を採用し、入力解像度への依存をなくした。


### darknet19の訓練

以下のコマンドでdarknet19のモデルの学習を行う。
変更点は、最終層に`softmax_cross_entropy`を使っている点以外、オリジナルコードと同じ。
`setup.sh`は、必要なディレクトリを作成する初期化スクリプト。

```
./setup.sh
python darknet19_train.py
```


### darknet19のテスト
以下のコマンドで、学習済みのモデルを使って、指定した画像をカテゴライズする。
テストに使用する学習済み重みのファイル等、ハイパーパラメータを変更する時は、`darknet19_predict.py`を書き換える。

```
python darknet19_predict.py "画像へのパス"
```