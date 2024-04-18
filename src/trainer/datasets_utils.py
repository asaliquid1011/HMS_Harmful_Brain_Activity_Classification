# ===============
# Import
# ===============
import numpy as np
import torch

###################
# Augmentation
###################
def flip_right_left(data,num_channels=16):
    #[[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]　→　[[1,0,2,3], [5,4,6,7], [9,8,10,11], [13,12,14,15]]
    result           = np.array(data)
    flip_num_channel = num_channels//2
    if data.ndim == 2:
        result[:flip_num_channel, :]        = data[flip_num_channel:, :]
        result[flip_num_channel:, :]        = data[:flip_num_channel, :]
    elif data.ndim == 3:
        result[:flip_num_channel, :, :]     = data[flip_num_channel:, :, :]
        result[flip_num_channel:, :, :]     = data[:flip_num_channel, :, :]
    elif data.ndim == 4:
        result[:flip_num_channel, :, :, :]  = data[flip_num_channel:, :, :, :]
        result[flip_num_channel:, :, :, :]  = data[:flip_num_channel, :, :, :]
    return result

def randomize_channels_in_range(data, ranges):
    result           = np.array(data)
    for range_list in ranges:
        range_list   = np.array(range_list)
        random_order = np.random.permutation(range_list)
        if data.ndim == 2:
            result[range_list, :]    = data[random_order, :]
        elif data.ndim == 3:
            result[range_list, :, :] = data[random_order, :, :]
        elif data.ndim == 4:
            result[range_list, :, :, :] = data[random_order, :, :, :]
    return result

def randomize_channels_in_range_torch(data, ranges):
    result              = data.clone()  # 元のデータを複製
    for range_list in ranges:
        range_tensor    = torch.tensor(range_list)
        # 指定された範囲内でランダムな順序を生成
        random_order    = torch.randperm(len(range_tensor))
        # ランダムな順序でチャンネルを並び替え
        if data.ndim == 3:
            result[:,range_tensor, :]         = data[:,range_tensor[random_order], :]
        elif data.ndim == 4:
            result[:,range_tensor, :, :]      = data[:,range_tensor[random_order], :, :]
        elif data.ndim == 5:
            result[:,range_tensor, :, :, :]   = data[:,range_tensor[random_order], :, :, :]
    return result

