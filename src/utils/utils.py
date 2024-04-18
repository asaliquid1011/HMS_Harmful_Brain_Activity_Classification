# -*- coding: utf-8 -*-
import sys
sys.path.append('../')


import os
import time
import math
import random
import pickle
import psutil
import torch
import numpy as np
import pandas as pd
from contextlib import contextmanager
import sys
sys.path.append('./')
from utils.logger import setup_logger, LOGGER

# ===============
# file操作
# ===============
def load_json(path):
    return pd.read_json(path, lines=True)

def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass

# ===============
# pickle
# ===============
def pickle_dump(obj, path):
    with open(path, mode='wb') as f:
        pickle.dump(obj,f)

def pickle_load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data

# ===============
# seed
# ===============
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# ===============
# config
# ===============
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = AttrDict(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = AttrDict(value)
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"No such attribute: {key}")
        
def replace_placeholders(data, placeholder_values):
    """
    データ内のプレースホルダーを特定の値で置き換える再帰関数。
    数値での置き換えに対応。
    """
    if isinstance(data, dict):
        return {k: replace_placeholders(v, placeholder_values) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_placeholders(v, placeholder_values) for v in data]
    elif isinstance(data, str):
        for key, value in placeholder_values.items():
            placeholder = '${' + key + '}'
            if placeholder in data:
                return value  # 数値として置き換える
        return data
    else:
        return data
    
# ===============
# time
# ===============
@contextmanager
def timer(name):
    t0 = time.time()
    yield
    LOGGER.info('[{}] done in {} s'.format(name, round(time.time() - t0, 2)))


# ===============
# Memory
# ===============
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

# ===============
# size
# ===============
def deep_getsizeof(obj, seen=None):
    # 既に探索済みのオブジェクトを保存するための集合
    if seen is None:
        seen = set()
    # objのIDを取得
    obj_id = id(obj)
    # オブジェクトが既に探索済みの場合は0を返す
    if obj_id in seen:
        return 0
    # オブジェクトIDを探索済みとして追加
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    # 辞書の場合、キーと値の両方を再帰的に探索
    if isinstance(obj, dict):
        size += sum([deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in obj.items()])
    # リスト、タプル、セットなどのコレクションの場合、要素を再帰的に探索
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([deep_getsizeof(item, seen) for item in obj])
    return size

# ===============
# time
# ===============
import time
def tic():
    #require to import time
    global start_time_tictoc
    start_time_tictoc = time.time()


def toc(tag="elapsed time"):
    if "start_time_tictoc" in globals():
        print("{}: {:.9f} [sec]".format(tag, time.time() - start_time_tictoc))
    else:
        print("tic has not been called")
        

@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)


def pad_if_needed(x: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    if len(x) == max_len:
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths, mode="constant", constant_values=pad_value)
    
# import sys
# print("{}{: >25}{}{: >10}{}".format('|','Variable Name','|','Memory','|'))
# print(" ------------------------------------ ")
# for var_name in dir():
#     if not var_name.startswith("_") and sys.getsizeof(eval(var_name)) > 100000: #ここだけアレンジ
#         print("{}{: >25}{}{: >10}{}".format('|',var_name,'|',sys.getsizeof(eval(var_name)),'|'))

