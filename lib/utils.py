import chainer.functions as F
import numpy as np
from chainer import Variable

def print_cnn_info(name, link, shape_before, shape_after, time):
    n_stride = (
        int((shape_before[2] + link.pad[0] * 2 - link.ksize) / link.stride[0]) + 1,
        int((shape_before[3] + link.pad[1] * 2 - link.ksize) / link.stride[1]) + 1
    )

    cost = n_stride[0] * n_stride[1] * shape_before[1] * link.ksize * link.ksize * link.out_channels

    print('%s(%d × %d, stride=%d, pad=%d) (%d x %d x %d) -> (%d x %d x %d) (cost=%d): %.6f[sec]' % 
        (
            name, link.W.shape[2], link.W.shape[3], link.stride[0], link.pad[0],
            shape_before[2], shape_before[3], shape_before[1], shape_after[2], shape_after[3], shape_after[1],
            cost, time
        )
    )

    return cost

def print_pooling_info(name, filter_size, stride, pad, shape_before, shape_after, time):
    n_stride = (
        int((shape_before[2] - filter_size) / stride) + 1,
        int((shape_before[3] - filter_size) / stride) + 1
    )
    cost = n_stride[0] * n_stride[1] * shape_before[1] * filter_size * filter_size * shape_after[1]

    print('%s(%d × %d, stride=%d, pad=%d) (%d x %d x %d) -> (%d x %d x %d) (cost=%d): %.6f[sec]' % 
        (name, filter_size, filter_size, stride, pad, shape_before[2], shape_before[3], shape_before[1], shape_after[2], shape_after[3], shape_after[1], cost, time)
    )

    return cost

def print_fc_info(name, link, time):
    import pdb
    cost = link.W.shape[0] * link.W.shape[1]
    print('%s %d -> %d (cost = %d): %.6f[sec]' % (name, link.W.shape[1], link.W.shape[0], cost, time))

    return cost

# x, y, w, hの4パラメータを保持するだけのクラス
class Box():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

# 2本の線の情報を受取り、被ってる線分の長さを返す。あくまで線分
def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

# chainerのVariable用のoverlap
def multi_overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = F.maximum(x1 - len1_half, x2 - len2_half)
    right = F.minimum(x1 + len1_half, x2 + len2_half)

    return right - left

# 2つのboxを受け取り、被ってる面積を返す(intersection of 2 boxes)
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

# chainer用
def multi_box_intersection(a, b):
    w = multi_overlap(a.x, a.w, b.x, b.w)
    h = multi_overlap(a.y, a.h, b.y, b.h)
    zeros = Variable(np.zeros(w.shape, dtype=w.data.dtype))
    zeros.to_gpu()

    w = F.maximum(w, zeros)
    h = F.maximum(h, zeros)

    area = w * h
    return area

# 2つのboxを受け取り、合計面積を返す。(union of 2 boxes)
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

# chianer用
def multi_box_union(a, b):
    i = multi_box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

# compute iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

# chainer用
def multi_box_iou(a, b):
    return multi_box_intersection(a, b) / multi_box_union(a, b)
