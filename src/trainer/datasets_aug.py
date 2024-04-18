# ===============
# Import
# ===============
import random
import numpy as np
import torch
import importlib

# ===============
# import mycode
# ===============
import sys
sys.path.append('../')
import trainer.datasets_utils
importlib.reload(trainer.datasets_utils)
#utils
from trainer.datasets_utils import randomize_channels_in_range,flip_right_left

# ==============================
# Augmentation_spec
# ==============================
def augmentation_spec(sub_spectrogram_4ch,phase,
                        img_trans_spec,
                        random_channel_spec,
                        random_ranges_spec,
                        resize_spec,
                        prob_spec):
                        
    if img_trans_spec is not None:
        sub_spectrogram_4ch             = img_trans_spec(sub_spectrogram_4ch.unsqueeze(0))
        sub_spectrogram_4ch             = sub_spectrogram_4ch.squeeze(0)

    #=other augmentation=#
    if (random_channel_spec!='off')&(phase == 'train'):
        if random.random()<prob_spec:
            if random_channel_spec=='random':
                random_order            = np.random.permutation(sub_spectrogram_4ch.shape[0])
                sub_spectrogram_4ch     = sub_spectrogram_4ch[random_order, :, :]
            elif random_channel_spec=='random_in_range':
                sub_spectrogram_4ch     = torch.tensor(randomize_channels_in_range(sub_spectrogram_4ch, random_ranges_spec))
        #両方
        if random_channel_spec=='random_in_range_flip_right_left':
            if random.random()<prob_spec:
                sub_spectrogram_4ch     = torch.tensor(randomize_channels_in_range(sub_spectrogram_4ch, random_ranges_spec))
            if random.random()<prob_spec:
                sub_spectrogram_4ch     = torch.tensor(flip_right_left(sub_spectrogram_4ch))
    #===resize===#
    if resize_spec is not None:
        sub_spectrogram_4ch             = resize_spec(sub_spectrogram_4ch)
    return sub_spectrogram_4ch

# ==============================
# Augmentation_eeg_wave
# ==============================
def augmentation_eeg_wave(sub_eeg_data_wave,phase,
                          wave_trans_eeg,sr_eeg_org,
                          wave_inverse,
                          random_channel_eeg_wave,
                          random_ranges_eeg_wave,
                          prob_eeg_wave):

    #=augmentation(wave)=#
    if wave_trans_eeg is not None:
        sub_eeg_data_wave               = wave_trans_eeg(sub_eeg_data_wave,sr_eeg_org)
    #=other augmentation=#
    #1.上下反転
    if (wave_inverse)&(phase == 'train'):
        if random.random()<0.5:
            sub_eeg_data_wave           = -1*sub_eeg_data_wave
    #4.random_channel_eeg_wave
    if (random_channel_eeg_wave!='off')&(phase == 'train'):
        if random.random()<prob_eeg_wave:
            if random_channel_eeg_wave=='random':
                random_order            = np.random.permutation(sub_eeg_data_wave.shape[0])
                sub_eeg_data_wave       = sub_eeg_data_wave[random_order, :]
            elif random_channel_eeg_wave=='random_in_range':
                sub_eeg_data_wave       = randomize_channels_in_range(sub_eeg_data_wave, random_ranges_eeg_wave)
        #両方
        if random_channel_eeg_wave=='random_in_range_flip_right_left':
            if random.random()<prob_eeg_wave:
                sub_eeg_data_wave       = randomize_channels_in_range(sub_eeg_data_wave, random_ranges_eeg_wave)
            if random.random()<prob_eeg_wave:
                sub_eeg_data_wave       = flip_right_left(sub_eeg_data_wave)
    sub_eeg_data_wave                   = np.nan_to_num(sub_eeg_data_wave)

    return sub_eeg_data_wave

# ==============================
# Augmentation_eeg_img
# ==============================
def augmentation_eeg_img(sub_eeg_data_img,phase,
                            img_trans_eeg,
                            random_channel_eeg_img,
                            random_ranges_eeg_img,
                            resize_eeg,
                            prob_eeg_img):                    
    #=augmentation(img)=#
    if img_trans_eeg is not None:
        sub_eeg_data_img                = img_trans_eeg(sub_eeg_data_img.unsqueeze(0))
        sub_eeg_data_img                = sub_eeg_data_img.squeeze(0)
    
    #random_channel_eeg_img
    if (random_channel_eeg_img!='off')&(phase == 'train'):
        if random.random()<prob_eeg_img:
            if random_channel_eeg_img=='random':
                random_order            = np.random.permutation(sub_eeg_data_img.shape[0])
                sub_eeg_data_img        = sub_eeg_data_img[random_order, :,:,:]
            elif random_channel_eeg_img=='random_in_range':
                sub_eeg_data_img        = torch.tensor(randomize_channels_in_range(sub_eeg_data_img, random_ranges_eeg_img))
        #両方
        if random_channel_eeg_img=='random_in_range_flip_right_left':
            if random.random()<prob_eeg_img:
                sub_eeg_data_img        = torch.tensor(randomize_channels_in_range(sub_eeg_data_img, random_ranges_eeg_img))
            if random.random()<prob_eeg_img:
                sub_eeg_data_img        = torch.tensor(flip_right_left(sub_eeg_data_img)) 
    #===resize===#
    if resize_eeg is not None:
        sub_eeg_data_img        = resize_eeg(sub_eeg_data_img)
    return sub_eeg_data_img 

