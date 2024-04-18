# -*- coding: utf-8 -*-
# ===============
# Import
# ===============
import sys
import torch
import torch.nn as nn
from torchvision import transforms
import timm
sys.path.append("..")

# ==============================
# CNN
# ==============================
class HMSModel_CNN(nn.Module):
    def __init__(self, config,input_type,output_type='logit'):
        super().__init__()
        #===config===#
        self.input_type             = input_type
        self.output_type            = output_type   #'logit' or 'encoder_output' or 'decoder_output'
        self.config                 = config
        if input_type =='SPEC':
            #deta setting
            n_classes               = config['dataset']['SPEC']['out_chans_spec']
            in_chans                = config['dataset']['SPEC']['in_chans_spec']
            self.set_in_chans       = in_chans
            in_chans_2              = config['dataset']['SPEC']['in_chans_spec_2']
            self.in_chans_2         = in_chans_2
            #model setting
            self.ch_mode            = config['model']['CNN_SPEC']['ch_mode'] 
            self.feature_extract    = config['model']['CNN_SPEC']['feature_extract']
            self.backbone_name      = config['model']['CNN_SPEC']['backbone_name_cnn']
            self.decoder            = config['model']['CNN_SPEC']['decoder']
            self.ch_head            = config['model']['CNN_SPEC']['ch_head']
            decoder_outclass        = config['model']['CNN_SPEC']['decoder_outclass']
        elif input_type =='EEG_IMG':
            #deta setting
            n_classes               = config['dataset']['EEG']['out_chans_eeg']
            in_chans                = config['dataset']['EEG']['in_chans_eeg_img']
            self.set_in_chans       = in_chans
            in_chans_2              = config['dataset']['EEG']['in_chans_eeg_img2']
            self.in_chans_2         = in_chans_2
            #model setting
            self.ch_mode            = config['model']['CNN_EEG']['ch_mode'] 
            self.feature_extract    = config['model']['CNN_EEG']['feature_extract']
            self.backbone_name      = config['model']['CNN_EEG']['backbone_name_cnn']
            self.decoder            = config['model']['CNN_EEG']['decoder']
            self.ch_head            = config['model']['CNN_EEG']['ch_head']
            decoder_outclass        = config['model']['CNN_EEG']['decoder_outclass']

        #====ch mode====#
        if self.ch_mode == '1ch':
            in_chans_def            = in_chans
            in_chans                = in_chans_2
            in_chans_decoder        = decoder_outclass
        else:
            in_chans_def            = 1
            in_chans                = in_chans
            in_chans_decoder        = n_classes
            
        #====Encoder====#
        #=backbone=#
        if self.decoder == 'none':
            self.backbone       = timm.create_model(
                                                self.backbone_name,
                                                pretrained          = True,
                                                num_classes         = in_chans_decoder,
                                                in_chans            = in_chans,
                                                # global_pool         = 'max',
                                            )
        else:
            self.backbone       = timm.create_model(
                                                self.backbone_name,
                                                pretrained          = True,
                                                num_classes         = 0,
                                                global_pool         = "",
                                                in_chans            = in_chans,
                                                )
            #backbone_out
            if "efficientnet"   in self.backbone_name:
                backbone_out        = self.backbone.num_features
            elif "rexnet"       in self.backbone_name:
                backbone_out        = self.backbone.num_features
            else:
                backbone_out        = self.backbone.feature_info[-1]["num_chs"]
            out_chans_decoder       = in_chans_decoder
            in_chans_decoder        = backbone_out

        #====Decoder ====#
        if self.decoder == 'none':
            #===None Decoder===#
            in_chans_head           = in_chans_decoder
        elif self.decoder == 'pooling_mlp':
            #===pooling_mlphead===#
            #setting
            if input_type =='SPEC':
                self.Global_pooling = config["model"]['CNN_SPEC']['pooling_mlp']['Global_pooling']
            elif input_type =='EEG_IMG':
                self.Global_pooling = config["model"]['CNN_EEG']['pooling_mlp']['Global_pooling']
            #pooling 
            if self.Global_pooling  == 'max':
                self.global_pool    = nn.AdaptiveMaxPool2d((1, 1))
            elif self.Global_pooling  == 'ave':
                self.global_pool    = nn.AdaptiveAvgPool2d((1, 1))
            elif self.Global_pooling  == 'max_ave':
                self.global_pool_ave= nn.AdaptiveAvgPool2d((1, 1))
                self.global_pool_max= nn.AdaptiveMaxPool2d((1, 1))
                in_chans_decoder      = in_chans_decoder*2
            #head
            self.head_mlp           = nn.Sequential(
                                            nn.Linear(in_chans_decoder , out_chans_decoder),
                                            )
            in_chans_head           = out_chans_decoder
        elif self.decoder == 'pass':
            pass
        
        #====Channel head ====#  
        if self.ch_mode == '1ch':
            #===Channel Head===#
            #=MLP Head=#
            if self.ch_head=='mlp':
                in_chans_head               = in_chans_def*in_chans_head
                #setting
                if input_type =='SPEC':
                    hidden_mlphead_ch       = config["model"]['CNN_SPEC']['mlp_head_ch']['hidden_mlphead_ch']
                    dropout_mlphead_ch      = config["model"]['CNN_SPEC']['mlp_head_ch']['dropout_mlphead_ch']
                elif input_type =='EEG_IMG':
                    hidden_mlphead_ch       = config["model"]['CNN_EEG']['mlp_head_ch']['hidden_mlphead_ch']
                    dropout_mlphead_ch      = config["model"]['CNN_EEG']['mlp_head_ch']['dropout_mlphead_ch']
                #head
                self.head_ch_mlp            = nn.Sequential(
                                                nn.Linear(in_chans_head, hidden_mlphead_ch),
                                                nn.LayerNorm(hidden_mlphead_ch),
                                                nn.ReLU(),
                                                nn.Dropout(dropout_mlphead_ch),
                                                nn.Linear(hidden_mlphead_ch, hidden_mlphead_ch),
                                                nn.LayerNorm(hidden_mlphead_ch),
                                                nn.ReLU(),
                                                nn.Dropout(dropout_mlphead_ch),
                                                nn.Linear(hidden_mlphead_ch, n_classes)
                                                )
            #=CNN2D Head=#
            elif self.ch_head=='cnn2d':
                #=setting=#
                if input_type =='SPEC':
                    self.backbone_name      = config['model']['CNN_SPEC']['cnn2d_head_ch']['backbone_name_head_ch']
                elif input_type =='EEG_IMG':
                    self.backbone_name      = config['model']['CNN_EEG']['cnn2d_head_ch']['backbone_name_head_ch']
                #=backbone=# 
                self.backbone_ch_head   = timm.create_model(
                                                    self.backbone_name,
                                                    pretrained          = True,
                                                    num_classes         = 6,
                                                    in_chans            = 1,
                                                    )
                self.img_size               = (256, 256)
                self.resize_torch           = transforms.Resize(self.img_size)

    #================================================================
    def forward(self, input):
        #===Pre_Process====#
        if self.input_type=='SPEC':
            bs, ch, h,w             = input.shape
            if (self.ch_mode == '1ch'):
                input               = input.reshape(bs*ch,1, h,w) #bs,ch, w,h→bs*ch,1, w,h
            else:
                pass
        elif self.input_type=='EEG_IMG':
            bs, ch, ch2, h,w        = input.shape
            if (self.ch_mode == '1ch'):
                input               = input.reshape(bs*ch,ch2, h,w)#bs,ch, w,h→bs*ch,1, w,h
            else:
                input               = input.reshape(bs,ch*ch2, h,w)#bs,ch, w,h→bs*ch,1, w,h


        #===encoder===#
        logits                      = self.backbone(input) #train(192, 2048, 4, 5) #valid(32, 2048, 16, 4))
        #=return1=#
        if self.output_type == 'encoder_output':
            return (logits.softmax(dim=1), logits)
        
        #===decoder===#
        if self.decoder == 'none':
            pass
        elif self.decoder == 'pooling_mlp':
            if self.Global_pooling  == 'max_ave':
                logits              = torch.cat([self.global_pool_ave(logits), self.global_pool_max(logits)], dim=1)
            else:
                logits              = self.global_pool(logits)
            logits                  = logits[:, :, 0, 0]
            logits                  = self.head_mlp(logits)
        elif self.decoder == 'pass':
            pass

        #=return2=#
        if self.output_type == 'decoder_output':
            return (logits.softmax(dim=1), logits)
        
        #===channel head===#
        if self.ch_mode == '1ch':
            #=head=#
            if self.ch_head=='mlp': 
                logits              = logits.reshape(bs, -1)
                logits              = self.head_ch_mlp(logits)
            elif self.ch_head=='cnn2d':
                bs_new,ch_cnn,h_cnn,w_cnn = logits.shape

                if self.input_type=='SPEC':
                    logits              = logits.reshape(bs_new,ch_cnn,h_cnn*w_cnn)
                    logits              = logits.reshape(bs,ch,ch_cnn,h_cnn*w_cnn) # [16, 4, 1280, 16]
                    logits              = logits.view(bs, 4 ,64, ch_cnn//64,h_cnn*w_cnn).mean(3)  # [16, 4, 64, 16]
                    logits              = logits.reshape(bs,4*64,h_cnn*w_cnn)
                    logits              = logits.unsqueeze(1)
                    
                    logits              = self.resize_torch(logits)
                    logits              = self.backbone_ch_head(logits)
                elif self.input_type=='EEG_IMG':
                    logits              = logits.reshape(bs_new,ch_cnn,h_cnn*w_cnn)
                    logits              = logits.reshape(bs,ch,ch_cnn,h_cnn*w_cnn)
                    logits              = logits.view(bs, 4, ch//4 , ch_cnn,h_cnn*w_cnn).mean(2)  # [16, 4, 1280, 16]
                    logits              = logits.view(bs, 4 ,64, ch_cnn//64,h_cnn*w_cnn).mean(3)  # [16, 4, 64, 16]
                    logits              = logits.reshape(bs,4*64,h_cnn*w_cnn)
                    logits              = logits.unsqueeze(1)
                    
                    logits              = self.resize_torch(logits)
                    logits              = self.backbone_ch_head(logits)
        else:
            pass

        return (logits.softmax(dim=1), logits)
