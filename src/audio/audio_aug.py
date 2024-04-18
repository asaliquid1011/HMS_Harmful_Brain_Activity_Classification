# -*- coding: utf-8 -*-
import numpy as np
import librosa
import random
# ===========================================
# # =get_trainsform=
# # ===========================================
class TimeMasking:
    def __init__(self, time_drop_width, time_stripes_num):
        self.time_drop_width = time_drop_width
        self.time_stripes_num = time_stripes_num

    def __call__(self, data,sr):
        # data形状: (ch, length)
        ch, length = data.shape
        for _ in range(self.time_stripes_num):
            t = np.random.randint(0, max(1, length - self.time_drop_width))
            data[:, t:t + self.time_drop_width] = 0
        return data


###################################################
def get_spectrogram_transforms(config,prob,phase):#torchに対して実行
    if phase=='train':
        from torchlibrosa.augmentation import SpecAugmentation
        import torchvision.transforms as T
        from torchvision.transforms import GaussianBlur, RandomApply, ColorJitter
        spec_trans          = config['aug_type']
        aug_list            = []
        if spec_trans=='time_freq_drop':
            aug_list.append(SpecAugmentation(time_drop_width    =config['time_drop_width'],#(bs,ch,height,width)の3軸目に対して適用される→timeと書いてるがfreqに適用
                                                 time_stripes_num   =config['time_stripes_num'],
                                                 freq_drop_width    =config['freq_drop_width'],#(bs,ch,height,width)の4軸目に対して適用される→freqと書いてるがtimeに適用
                                                 freq_stripes_num   =config['freq_stripes_num']))
            transforms = T.Compose(aug_list)

        elif spec_trans=='HorizontalFlip':
            aug_list.append(T.RandomHorizontalFlip(p=0.5))
            transforms = T.Compose(aug_list)
            
        elif spec_trans=='time_freq_drop_HorizontalFlip':
            aug_list.append(T.RandomHorizontalFlip(p=0.5))
            aug_list.append(SpecAugmentation(time_drop_width        =config['time_drop_width'],
                                                 time_stripes_num   =config['time_stripes_num'],
                                                 freq_drop_width    =config['freq_drop_width'],
                                                 freq_stripes_num   =config['freq_stripes_num']))
            transforms      = T.Compose(aug_list)
        elif spec_trans=='tf_drop_HFlip_blur':
            aug_list.append(T.RandomHorizontalFlip(p=0.5))
            aug_list.append(SpecAugmentation(time_drop_width        =config['time_drop_width'],
                                                 time_stripes_num   =config['time_stripes_num'],
                                                 freq_drop_width    =config['freq_drop_width'],
                                                 freq_stripes_num   =config['freq_stripes_num']))
            aug_list.append(RandomApply([GaussianBlur(kernel_size=config['kernel_size_blur'])], p=0.5))
            transforms      = T.Compose(aug_list)
        else:
            transforms      =   None
    else:#valid
        transforms          =   None
    return transforms

###################################################
def get_wave_transforms(config,prob,phase):#numpyに対して実行
    if phase =='train':
        from audiomentations import Compose, AddGaussianSNR, AddGaussianNoise, PitchShift, TimeStretch
        wave_trans          = config['aug_type']
        aug_list            = []
        if wave_trans=='GaussSNR':
            # AddGaussianSNR: ガウシアンSNRノイズを追加（デフォルト: min_SNR=10, max_SNR=50, p=0.5）
            aug_list.append(AddGaussianSNR(min_snr_db=config['gauss_snr_min'], 
                                           max_snr_db=config['gauss_snr_max'], p=prob))
            transforms      = Compose(aug_list)
        elif wave_trans=='GaussNoise':
            # AddGaussianNoise: ガウシアンノイズを追加（デフォルト: min_amplitude=0.0001, max_amplitude=0.001, p=0.5）
            aug_list.append(AddGaussianNoise(min_amplitude=config['gauss_noise_min'],
                                             max_amplitude=config['gauss_noise_max'], p=prob))
            transforms      = Compose(aug_list)
        elif wave_trans=='PinkNoise':
            # AddPinkNoise: ピンクノイズを追加（デフォルト: min_amplitude=0.0001, max_amplitude=0.001, p=0.5）
            aug_list.append(AddPinkNoise(min_amplitude=0.0001, max_amplitude=0.001, p=prob))
            transforms      = Compose(aug_list)
        elif wave_trans=='PitchShift':
            # PitchShift: ピッチシフト（デフォルト: min_semitones=-4, max_semitones=4, sample_rate=44100, p=0.5）
            aug_list.append(PitchShift(min_semitones=-4, max_semitones=4, sample_rate=44100, p=prob))
            transforms      = Compose(aug_list)
        elif wave_trans=='TimeStretch':
            # TimeStretch: タイムストレッチ（デフォルト: min_rate=0.8, max_rate=1.25, p=0.5）
            aug_list.append(TimeStretch(min_rate=0.8, max_rate=1.25, p=prob))
            transforms      = Compose(aug_list)
        # カスタム時間方向マスキングの追加
        elif wave_trans == 'TimeMask':
            aug_list.append(TimeMasking(time_drop_width     = config['time_drop_width'],
                                        time_stripes_num    = config['time_stripes_num']))
            transforms = Compose(aug_list)
        else:
            transforms = None
    else:#valid
        transforms          =   None
    return transforms
