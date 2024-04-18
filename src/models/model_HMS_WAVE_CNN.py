# -*- coding: utf-8 -*-
# ===============
# Import
# ===============
import math
import torch
import torch.nn as nn
from torchvision import transforms
import timm

# ==============================
# Wave_IMG
# ==============================
class HMSModel_WAVE_CNN(torch.nn.Module):
    def __init__(self, config,output_type='logit'):
        super().__init__()
        #===config===#
        self.output_type                = output_type   #'logit' or 'encoder_output' or 'decoder_output'
        self.config                     = config
        self.n_classes                  = config['dataset']['EEG']['out_chans_eeg']
        in_chans                        = config['dataset']['EEG']['in_chans_eeg_wave']
        self.ch_mode                    = config['model']['WAVE_EEG_CNN']['ch_mode'] #1ch or 16ch
        if self.ch_mode == "1ch":
            in_chans_def                = in_chans
            in_chans                    = 1
            num_groups                  = 1
        else:
            in_chans_def                = 1
            num_groups                  = in_chans

        #====setting====#
        self.feature_extract            = config['model']['WAVE_EEG_CNN']['feature_extract']
        self.encoder                    = config['model']['WAVE_EEG_CNN']['encoder']
        self.decoder                    = config['model']['WAVE_EEG_CNN']['decoder']
        self.ch_head                    = config['model']['WAVE_EEG_CNN']['ch_head']
        self.epoch_change_decoder       = config['model']['WAVE_EEG_CNN']['epoch_change_decoder']
        self.flg_change_decoder_pooling2cnn = False

        #====Feature Extractor====#
        if self.feature_extract=='multi_1dcnn':
            #====multi_1dcnn====#
            multi_channels              = config['model']['WAVE_EEG_CNN']['multi_1dcnn']['multi_channels']
            multi_kernel_sizes          = config['model']['WAVE_EEG_CNN']['multi_1dcnn']['multi_kernel_sizes']
            multi_strides               = config['model']['WAVE_EEG_CNN']['multi_1dcnn']['multi_strides']
            multi_res_depth             = config['model']['WAVE_EEG_CNN']['multi_1dcnn']['res_depth']
            multi_se_ratio              = config['model']['WAVE_EEG_CNN']['multi_1dcnn']['se_ratio']
            multi_res_kernel_size       = config['model']['WAVE_EEG_CNN']['multi_1dcnn']['res_kernel_size']
            self.layer_multi_1dcnn      = nn.ModuleList()
            sum_chan                    = 0
            for i in range(len(multi_channels)):
                out_chan                = multi_channels[i]
                kernel_size             = multi_kernel_sizes[i]
                stride                  = multi_strides[i]
                sum_chan                += out_chan
                block                   = []
                block.append(ConvBNReLU(in_chans, out_chan, kernel_size, stride,groups=num_groups))
                for j in range(multi_res_depth):#3
                    block.append(ResBlock(out_chan, multi_res_kernel_size, multi_se_ratio))
                self.layer_multi_1dcnn.append(nn.Sequential(*block))
            #更新
            in_chans_encoder            = sum_chan
        else:
            in_chans_encoder            = in_chans

        #====Encoder ====#
        if self.encoder == 'deep_1dcnn':
            #====deep_1dcnn====#
            down_channels               = config['model']['WAVE_EEG_CNN']['deep_1dcnn']['down_channels']
            down_kernel_size            = config['model']['WAVE_EEG_CNN']['deep_1dcnn']['down_kernel_size']
            down_stride                 = config['model']['WAVE_EEG_CNN']['deep_1dcnn']['down_stride']
            res_depth                   = config['model']['WAVE_EEG_CNN']['deep_1dcnn']['res_depth']
            se_ratio                    = config['model']['WAVE_EEG_CNN']['deep_1dcnn']['se_ratio']
            res_kernel_size             = config['model']['WAVE_EEG_CNN']['deep_1dcnn']['res_kernel_size']
            self.layer_deep_1dcnn       = nn.ModuleList()
            for i in range(len(down_channels)):
                if i == 0:
                    in_chan_1dcnn       = in_chans_encoder
                else:
                    # in_channels         = down_channels[i-1] + in_channels_2nd 
                    in_chan_1dcnn       = down_channels[i-1]
                out_chan_1dcnn          = down_channels[i]
                kernel_size             = down_kernel_size[i]
                stride                  = down_stride[i]
                block                   = []
                block.append(ConvBNReLU(in_chan_1dcnn, out_chan_1dcnn, kernel_size, stride,groups=num_groups))
                for j in range(res_depth):#3
                    block.append(ResBlock(out_chan_1dcnn, res_kernel_size, se_ratio))
                self.layer_deep_1dcnn.append(nn.Sequential(*block))
            #更新
            in_chans_decoder            = down_channels[-1]* 1
        
        #====Dencoder ====#
        if (self.decoder == 'pooling'):
            #===pooling_MLPhead===#
            #=time_pooling
            self.mode_time_pooling      = config['model']['WAVE_EEG_CNN']['pooling']['mode_time_pooling']
            if self.mode_time_pooling   =='max':
                self.time_pooling       = nn.AdaptiveMaxPool1d(1)
            elif self.mode_time_pooling =='avg':
                self.time_pooling       = nn.AdaptiveAvgPool1d(1)
            elif self.mode_time_pooling =='max_avg':
                self.time_pooling_max   = nn.AdaptiveMaxPool1d(1)
                self.time_pooling_avg   = nn.AdaptiveAvgPool1d(1)        
        else:
            pass
        
        #====Channel head ====# 
        if self.ch_head=='mlp':
            in_chans_ch_head            = in_chans_def*in_chans_ch_head
            #=MLP Head=#
            hidden_mlphead_ch           = config['model']['WAVE_EEG_CNN']['mlp_head_ch']['hidden_mlphead_ch']
            dropout_mlphead_ch          = config['model']['WAVE_EEG_CNN']['mlp_head_ch']['dropout_mlphead_ch']
            self.head_ch_mlp            = nn.Sequential(
                                                        nn.Linear(in_chans_ch_head,hidden_mlphead_ch),
                                                        nn.LayerNorm(hidden_mlphead_ch),
                                                        nn.ReLU(),
                                                        nn.Dropout(dropout_mlphead_ch),
                                                        nn.Linear(hidden_mlphead_ch,hidden_mlphead_ch),
                                                        nn.LayerNorm(hidden_mlphead_ch),
                                                        nn.ReLU(),
                                                        nn.Dropout(dropout_mlphead_ch),
                                                        nn.Linear(hidden_mlphead_ch, self.n_classes)
                                                        )
            
        elif self.ch_head=='cnn2d':
            #=setting=#
            self.backbone_name          = config['model']['WAVE_EEG_CNN']['cnn2d_head_ch']['backbone_name_head_ch']
            self.backbone_ch_head       = timm.create_model(
                                                self.backbone_name,
                                                pretrained          = True,
                                                num_classes         = 6,
                                                in_chans            = 1,
                                                )
            self.img_size               = (256, 256)
            self.resize_torch           = transforms.Resize(self.img_size)

            #=pooling=# 
            self.mode_img_pooling       = config['model']['WAVE_EEG_CNN']['cnn2d_head_ch']['mode_img_pooling']
            if self.mode_img_pooling   =='max':
                self.img_pooling        = nn.AdaptiveMaxPool2d((1, 1))
            elif self.mode_img_pooling =='avg':
                self.img_pooling        = nn.AdaptiveAvgPool2d((1, 1))
            elif self.mode_img_pooling =='max_avg':
                self.img_pooling_max    = nn.AdaptiveMaxPool2d((1, 1))
                self.img_pooling_avg    = nn.AdaptiveAvgPool2d((1, 1))            
                

    #================
    def forward(self, x,epoch=99999):
        #===Pre_Process====#
        #x                       = self.bn1(x)
        if self.ch_mode == "1ch":
            bs, channels, time_int      = x.shape
            x                           = x.reshape(bs*channels,-1) #(batch_size, channels,time_int)→(batch_size*channel,time)
            x                           = x.unsqueeze(1) #(batch_size*channel,time)→(batch_size*channel,1,time)

        #===Feature Extractor===#
        if self.feature_extract=='multi_1dcnn':
            #====multi_1dcnn====#
            out                         = []
            for i in range(len(self.layer_multi_1dcnn)):
                tmp                     = x
                for j in range(len(self.layer_multi_1dcnn[i])):
                    tmp                 = self.layer_multi_1dcnn[i][j](tmp)
                out.append(tmp) 
            x                           = torch.cat(out, dim=1)
        else:
            pass

        #===encoder===#
        if self.encoder == 'deep_1dcnn':
            #====deep_1dcnn====#
            for i in range(len(self.layer_deep_1dcnn)):
                for j in range(len(self.layer_deep_1dcnn[i])):
                    x                   = self.layer_deep_1dcnn[i][j](x) #(batch_size*channel,ch,time_圧縮)
        else:
            pass

        #=return1=#
        if self.output_type == 'encoder_output':
            return (x.softmax(dim=1), x)

        #====Decoder ====#
        if self.decoder == 'pooling':
            #====pooling====#
            #time pool
            if self.mode_time_pooling!='max_avg':
                x                   = self.time_pooling(x)
            elif self.mode_time_pooling=='max_avg':
                x                   = torch.cat([self.time_pooling_max(x), self.time_pooling_avg(x)], dim=1)
            x                       = x[:, :, 0]   #(batch_size,d_model)
        else:
            pass

        #=return2=#
        if self.output_type == 'decoder_output':
            return (x.softmax(dim=1), x)
        
        #=head=#
        if self.ch_head=='mlp': 
            x                   = x.reshape(bs,-1)
            x                   = self.head_ch_mlp(x)
        elif self.ch_head=='cnn2d':
            _,h,w       = x.shape
            x                   = x.reshape(bs,-1,h,w)
            x                   = x.view(bs, 4, channels//4 , h, w).mean(2)  # [32, 4, 64, 74]
            x                   = x.reshape(bs, 1, 4*h, w)  # [32, 1, 4*64, 74]
            x                   = self.resize_torch(x)
            x                   = self.backbone_ch_head(x)

        return (x.softmax(dim=1), x)
    
# ========================================================================================================================
# Blocks
# ========================================================================================================================
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()
        
        if stride == 1:
            padding = "same"#stride1の場合、入力と出力のsizeは変わらない
        else:
            # padding = (kernel_size - stride) // 2
            padding = math.ceil((kernel_size - stride) / 2)

        # print(out_channels)
        # print(groups)
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, 
                      padding=padding, groups=groups),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        x_out = self.layers(x)
        return x_out


class SEBlock(nn.Module):#size変えない
    def __init__(self, n_channels, se_ratio):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),  #  Global Average Pooling (bs,C,1)  
            nn.Conv1d(n_channels, n_channels//se_ratio, kernel_size=1),# (bs,C//se_ratio,1)  CH方向の圧縮
            nn.ReLU(),
            nn.Conv1d(n_channels//se_ratio, n_channels, kernel_size=1),# (bs,C,1)  CH方向の元に戻す
            nn.Sigmoid() #(bs,C,1)
        )
    def forward(self, x):
        x_out = torch.mul(x, self.layers(x))#  (bs,C,L) × (bs,C,1)。行列要素積self.layers(x)がチャンネルの重みみたいなもの
        return x_out

class ResBlock(nn.Module):#size変えない
    def __init__(self, n_channels, kernel_size, se_ratio):
        super().__init__()

        self.layers = nn.Sequential(
                        ConvBNReLU(n_channels, n_channels, kernel_size, stride=1),
                        ConvBNReLU(n_channels, n_channels, kernel_size, stride=1),
                        SEBlock(n_channels, se_ratio)
                    )
    def forward(self, x):
        x_re    = self.layers(x)
        x_out   = x + x_re
        return x_out

