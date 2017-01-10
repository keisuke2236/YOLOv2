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
