import math
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from nnAudio import Spectrogram as Spec
from torchlibrosa.augmentation import SpecAugmentation
import config
from models_panns import *

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)
    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
class T_Attention(nn.Module):
    def __init__(self, in_channels):
        
        super(T_Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,
                              kernel_size=(1, 1), stride=(1, 1),padding=(0, 0), bias=False)
                              
        self.bn = nn.BatchNorm2d(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv)
        init_bn(self.bn)

    def forward(self, input):
        
        x = input
        y = self.conv(input)
        res = x.transpose(1,2)
        y = y.transpose(1,2)
        y = self.pool(y)
        y = y.view(y.size(0), -1)
        y = self.sigmoid(y)
        y = y.view(y.size(0), y.size(1), 1, 1)
        y = y * res
        y = y.transpose(1,2)
        y = self.bn(self.relu(y))
        
        return y

class F_Attention(nn.Module):
    def __init__(self, in_channels):
        
        super(F_Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,out_channels=in_channels,
                              kernel_size=(1, 1), stride=(1, 1),padding=(0, 0), bias=False)
                              
        self.bn = nn.BatchNorm2d(in_channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv)
        init_bn(self.bn)

    def forward(self, input):
        
        x = input
        y = self.conv(input)
        res = x.transpose(1,3)
        y = y.transpose(1,3)
        y = self.pool(y)
        y = y.view(y.size(0), -1)
        y = self.sigmoid(y)
        y = 0.5 + 0.5*y
        y = y.view(y.size(0), y.size(1), 1, 1)
        y = y * res
        y = y.transpose(1,3)
        y = self.bn(self.relu(y))
        
        return y
    
class TFBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(TFBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                             out_channels=out_channels,
                                 kernel_size=(3, 3), stride=(1, 1),
                             padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.t_a = T_Attention(out_channels)
        self.f_a = F_Attention(out_channels)
        self.alpha = nn.Parameter(torch.cuda.FloatTensor([.1, .1, .1]))
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        y = self.t_a(x)
        z = self.f_a(x)
        so_alpha = F.softmax(self.alpha,dim=0)
        x = so_alpha[0]*x + so_alpha[1]*y + so_alpha[2]*z
        x = self.bn(x)
        #x = z
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
class Logmel_Cnn(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Logmel_Cnn, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=5, freq_stripes_num=2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x1
        x = self.spec_layer(x)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        #if self.training:
            #x = self.spec_augmenter(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Cqt_Cnn(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Cqt_Cnn, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.CQT(sr=config.sample_rate, hop_length=config.hop_size, fmin=220, fmax=None, n_bins=64, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect')
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=5, freq_stripes_num=2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x1
        x = self.spec_layer(x)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        #if self.training:
            #x = self.spec_augmenter(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Gamm_Cnn(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Gamm_Cnn, self).__init__()

        self.activation = activation
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=5, freq_stripes_num=2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x2
        x = x[:,None,:,:]
        #if self.training:
            #x = self.spec_augmenter(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Mfcc_Cnn(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Mfcc_Cnn, self).__init__()

        self.activation = activation
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=5, freq_stripes_num=2)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x3
        x = x[:,None,:,:]
        #if self.training:
            #x = self.spec_augmenter(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        # x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Ensemble_CNN(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Ensemble_CNN, self).__init__()

        self.activation = activation
        self.logmel_cnn = Logmel_Cnn(classes_num, activation, fixed=False)
        self.cqt_cnn = Cqt_Cnn(classes_num, activation, fixed=False)
        self.gamm_cnn = Gamm_Cnn(classes_num, activation, fixed=False)
        self.mfcc_cnn = Mfcc_Cnn(classes_num, activation, fixed=False)
        # self.ite = 15000 #12000
        self.ite = 15000 #1200
        # self.model_path = 'D:/Project/DCASE_test/checkpoints/liufeng/logmel_86frames_40melbins/TAU-urban-acoustic-scenes-2020-mobile-development/holdout_fold=1/'
        self.model_path = 'D:/Project/DCASE_test/checkpoints/liufeng/logmel_86frames_40melbins/TAU-urban-acoustic-scenes-2020-mobile-development/holdout_fold=1/'
        self.init_weights()

    def init_weights(self):
        logmel_path = self.model_path +'Logmel_Cnn/'+str(self.ite)+'_iterations.pth'
        cqt_path = self.model_path +'Cqt_Cnn/'+str(self.ite)+'_iterations.pth'
        gamm_path = self.model_path +'Gamm_Cnn/'+str(self.ite)+'_iterations.pth'
        mfcc_path = self.model_path +'Mfcc_Cnn/'+str(self.ite)+'_iterations.pth'

        logmel_ch = torch.load(logmel_path)
        cqt_ch = torch.load(cqt_path)
        gamm_ch = torch.load(gamm_path)
        mfcc_ch = torch.load(mfcc_path)
        self.logmel_cnn.load_state_dict(logmel_ch['model'])
        self.cqt_cnn.load_state_dict(cqt_ch['model'])
        self.gamm_cnn.load_state_dict(gamm_ch['model'])
        self.mfcc_cnn.load_state_dict(mfcc_ch['model'])

    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.logmel_cnn(input1,input2,input3,input4)
        x2,loss2 = self.cqt_cnn(input1,input2,input3,input4)
        x3,loss3 = self.gamm_cnn(input1,input2,input3,input4)
        x4,loss4 = self.mfcc_cnn(input1,input2,input3,input4)
        x = x1 + x2 + x3 + x4
        loss = (loss1+loss2+loss3+loss4)/4.
        return x,loss
    
class TFNet(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(TFNet, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        self.conv_block1 = TFBlock(in_channels=1, out_channels=64)
        self.conv_block2 = TFBlock(in_channels=64, out_channels=128)
        self.conv_block3 = TFBlock(in_channels=128, out_channels=256)
        self.conv_block4 = TFBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x1
        x = self.spec_layer(x)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss


class Logmel_Res38(nn.Module):
    def __init__(self, classes_num, activation, fixed=True):
        
        super(Logmel_Res38, self).__init__()

        self.activation = activation
        self.resnet38 = ResNet38(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527)
        
        self.init_model()
        if fixed==True:
            for p in self.parameters():
                p.requires_grad=False
                
        self.fc = nn.Linear(2048, classes_num, bias=True)
        self.init_weights()
        
    def init_model(self, model_path='/home/cdd/code2/dcase2020_task1/pytorch/ResNet38_mAP=0.434.pth'):
        device = torch.device('cuda')
        checkpoint = torch.load(model_path, map_location=device)
        self.resnet38.load_state_dict(checkpoint['model'])
                
    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x1,x2,x3,x4):
        """
        Input: (batch_size, data_length)"""
        x = self.resnet38(x4)
        x = x['embedding']
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss
        
class Logmel_Wavegram_Logmel_Cnn14(nn.Module):
    def __init__(self, classes_num, activation, fixed=True):
        
        super(Logmel_Wavegram_Logmel_Cnn14, self).__init__()

        self.activation = activation
        self.net = Wavegram_Logmel_Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527)
        
        self.init_model()
        if fixed==True:
            for p in self.parameters():
                p.requires_grad=False
                
        self.fc = nn.Linear(2048, classes_num, bias=True)
        self.init_weights()
        
    def init_model(self, model_path='/home/cdd/code2/dcase2020_task1/pytorch/Wavegram_Logmel_Cnn14_mAP=0.439.pth'):
        device = torch.device('cuda')
        checkpoint = torch.load(model_path, map_location=device)
        self.net.load_state_dict(checkpoint['model'])
                
    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x1,x2,x3,x4):
        """
        Input: (batch_size, data_length)"""
        x = self.net(x4)
        x = x['embedding']
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss
    
class Logmel_Cnn14(nn.Module):
    def __init__(self, classes_num, activation, fixed=True):
        
        super(Logmel_Cnn14, self).__init__()

        self.activation = activation
        self.net = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527)
        
        self.init_model()
        if fixed==True:
            for p in self.parameters():
                p.requires_grad=False
                
        self.fc = nn.Linear(2048, classes_num, bias=True)
        self.init_weights()
        
    def init_model(self, model_path='/home/cdd/code2/dcase2020_task1/pytorch/Cnn14_mAP=0.431.pth'):
        device = torch.device('cuda')
        checkpoint = torch.load(model_path, map_location=device)
        self.net.load_state_dict(checkpoint['model'])
                
    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x1,x2,x3,x4):
        """
        Input: (batch_size, data_length)"""
        x = self.net(x4)
        x = x['embedding']
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Logmel_Cnn10(nn.Module):
    def __init__(self, classes_num, activation, fixed=True):
        
        super(Logmel_Cnn10, self).__init__()

        self.activation = activation
        self.net = Cnn10(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527)
        
        self.init_model()
        if fixed==True:
            for p in self.parameters():
                p.requires_grad=False
                
        self.fc = nn.Linear(512, classes_num, bias=True)
        self.init_weights()
        
    def init_model(self, model_path='/home/cdd/code2/dcase2020_task1/pytorch/Cnn10_mAP=0.380.pth'):
        device = torch.device('cuda')
        checkpoint = torch.load(model_path, map_location=device)
        self.net.load_state_dict(checkpoint['model'])
                
    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x1,x2,x3,x4):
        """
        Input: (batch_size, data_length)"""
        x = self.net(x4)
        x = x['embedding']
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Logmel_Wavegram_Cnn14(nn.Module):
    def __init__(self, classes_num, activation, fixed=True):
        
        super(Logmel_Wavegram_Cnn14, self).__init__()

        self.activation = activation
        self.net = Wavegram_Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527)
        
        self.init_model()
        if fixed==True:
            for p in self.parameters():
                p.requires_grad=False
                
        self.fc = nn.Linear(2048, classes_num, bias=True)
        self.init_weights()
        
    def init_model(self, model_path='/home/cdd/code2/dcase2020_task1/pytorch/Wavegram_Cnn14_mAP=0.389.pth'):
        device = torch.device('cuda')
        checkpoint = torch.load(model_path, map_location=device)
        self.net.load_state_dict(checkpoint['model'])
                
    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x1,x2,x3,x4):
        """
        Input: (batch_size, data_length)"""
        x = self.net(x4)
        x = x['embedding']
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss
    
class Logmel_MobileNetV2(nn.Module):
    def __init__(self, classes_num, activation, fixed=True):
        
        super(Logmel_MobileNetV2, self).__init__()

        self.activation = activation
        self.net = MobileNetV2(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527)
        
        self.init_model()
        if fixed==True:
            for p in self.parameters():
                p.requires_grad=False
                
        self.fc = nn.Linear(1024, classes_num, bias=True)
        self.init_weights()
        
    def init_model(self, model_path='/home/cdd/code2/dcase2020_task1/pytorch/MobileNetV2_mAP=0.383.pth'):
        device = torch.device('cuda')
        checkpoint = torch.load(model_path, map_location=device)
        self.net.load_state_dict(checkpoint['model'])
                
    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x1,x2,x3,x4):
        """
        Input: (batch_size, data_length)"""
        x = self.net(x4)
        x = x['embedding']
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Logmel_MobileNetV1(nn.Module):
    def __init__(self, classes_num, activation, fixed=True):
        
        super(Logmel_MobileNetV1, self).__init__()

        self.activation = activation
        self.net = MobileNetV1(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50, 
        fmax=14000, classes_num=527)
        
        self.init_model()
        if fixed==True:
            for p in self.parameters():
                p.requires_grad=False
                
        self.fc = nn.Linear(1024, classes_num, bias=True)
        self.init_weights()
        
    def init_model(self, model_path='/home/cdd/code2/dcase2020_task1/pytorch/MobileNetV1_mAP=0.389.pth'):
        device = torch.device('cuda')
        checkpoint = torch.load(model_path, map_location=device)
        self.net.load_state_dict(checkpoint['model'])
                
    def init_weights(self):
        init_layer(self.fc)

    def forward(self, x1,x2,x3,x4):
        """
        Input: (batch_size, data_length)"""
        x = self.net(x4)
        x = x['embedding']
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss
    
class Ensemble_CNN2(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Ensemble_CNN2, self).__init__()

        self.activation = activation
        self.model1 = Logmel_Res38(classes_num, activation, fixed=False)
        self.model2 = Logmel_Wavegram_Logmel_Cnn14(classes_num, activation, fixed=False)
        self.model3 = Logmel_Cnn10(classes_num, activation, fixed=False)
        self.model4 = Logmel_Cnn14(classes_num, activation, fixed=False)
        self.model5 = Logmel_MobileNetV2(classes_num, activation, fixed=False)
        self.model6 = Logmel_MobileNetV1(classes_num, activation, fixed=False)
        self.model7 = Logmel_Wavegram_Cnn14(classes_num, activation, fixed=False)
        self.ite = 15000 #12000
        self.model_path = 'D:/Project/DCASE_test/checkpoints/liufeng/logmel_86frames_40melbins/TAU-urban-acoustic-scenes-2020-mobile-development/holdout_fold=1/'
        self.init_weights()

    def init_weights(self):
        model_path1 = self.model_path +'Logmel_Res38/'+str(self.ite)+'_iterations.pth'
        model_path2 = self.model_path +'Logmel_Wavegram_Logmel_Cnn14/'+str(self.ite)+'_iterations.pth'
        model_path3 = self.model_path +'Logmel_Cnn10/'+str(self.ite)+'_iterations.pth'
        model_path4 = self.model_path +'Logmel_Cnn14/'+str(self.ite)+'_iterations.pth'
        model_path5 = self.model_path +'Logmel_MobileNetV2/'+str(self.ite)+'_iterations.pth'
        model_path6 = self.model_path +'Logmel_MobileNetV1/'+str(self.ite)+'_iterations.pth'
        model_path7 = self.model_path +'Logmel_Wavegram_Cnn14/'+str(self.ite)+'_iterations.pth'

        model_ch1 = torch.load(model_path1)
        model_ch2 = torch.load(model_path2)
        model_ch3 = torch.load(model_path3)
        model_ch4 = torch.load(model_path4)
        model_ch5 = torch.load(model_path5)
        model_ch6 = torch.load(model_path6)
        model_ch7 = torch.load(model_path7)
        
        self.model1.load_state_dict(model_ch1['model'])
        self.model2.load_state_dict(model_ch2['model'])
        self.model3.load_state_dict(model_ch3['model'])
        self.model4.load_state_dict(model_ch4['model'])
        self.model5.load_state_dict(model_ch5['model'])
        self.model6.load_state_dict(model_ch6['model'])
        self.model7.load_state_dict(model_ch7['model'])
       

    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.model1(input1,input2,input3,input4)
        x2,loss2 = self.model2(input1,input2,input3,input4)
        x3,loss3 = self.model3(input1,input2,input3,input4)
        x4,loss4 = self.model4(input1,input2,input3,input4)
        x5,loss5 = self.model5(input1,input2,input3,input4)
        x6,loss6 = self.model6(input1,input2,input3,input4)
        x7,loss7 = self.model7(input1,input2,input3,input4)
        
        #x = x1 + x2 + x3 + x4 + x5 + x6 + x7 0.839
        
        x = x1 + x2 + x3 + x4 + x7
        loss = (loss1 + loss2 + loss3 + loss4 + loss7)/5.

#         x = x1 + x2 + x7
#         loss = (loss1 + loss2 + loss7)/3.
        return x,loss
    
class Logmel_MultiFrebands_CNN(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Logmel_MultiFrebands_CNN, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.fc1_1 = nn.Linear(512, 512, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc2_1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc3_1 = nn.Linear(512, 512, bias=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc4_1 = nn.Linear(512, 512, bias=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc5_1 = nn.Linear(512, 512, bias=True)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc5_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc6_1 = nn.Linear(512, 512, bias=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc6_2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1_1)
        init_layer(self.fc1_2)
        init_layer(self.fc2_1)
        init_layer(self.fc2_2)
        init_layer(self.fc3_1)
        init_layer(self.fc3_2)
        init_layer(self.fc4_1)
        init_layer(self.fc4_2)
        init_layer(self.fc5_1)
        init_layer(self.fc5_2)
        init_layer(self.fc6_1)
        init_layer(self.fc6_2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = input
        x = self.spec_layer(x1)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        temp = x.clone()
        
        count = 0.
        x = torch.mean(temp, dim=3)     # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc1_1(x))
        x = self.dropout1(x)
        x = self.fc1_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,0]
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2_1(x))
        x = self.dropout2(x)
        x = self.fc2_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,1]
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc3_1(x))
        x = self.dropout3(x)
        x = self.fc3_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,2]
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc4_1(x))
        x = self.dropout4(x)
        x = self.fc4_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,3]
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc5_1(x))
        x = self.dropout5(x)
        x = self.fc5_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,4]
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc6_1(x))
        x = self.dropout6(x)
        x = self.fc6_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        outputs.append(x)
        count += 1.
        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss

class Logmel_SubFrebands_CNN(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Logmel_SubFrebands_CNN, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        self.conv_block1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block1_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block1_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block1_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block2_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block2_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block2_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block3_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block3_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block4_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block4_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block4_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_4 = ConvBlock(in_channels=256, out_channels=512)
                              
                             
        self.fc1_1 = nn.Linear(512, 512, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc2_1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc3_1 = nn.Linear(512, 512, bias=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc4_1 = nn.Linear(512, 512, bias=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4_2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1_1)
        init_layer(self.fc1_2)
        init_layer(self.fc2_1)
        init_layer(self.fc2_2)
        init_layer(self.fc3_1)
        init_layer(self.fc3_2)
        init_layer(self.fc4_1)
        init_layer(self.fc4_2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = input
        x = self.spec_layer(x1)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        temp = x.clone()
        count = 0.
        
        x = temp[:,:,:,0:16]
        x = self.conv_block1_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block1_4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc1_1(x))
        x = self.dropout1(x)
        x = self.fc1_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        
        x = temp[:,:,:,8:24]
        x = self.conv_block2_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2_1(x))
        x = self.dropout2(x)
        x = self.fc2_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,16:32]
        x = self.conv_block3_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc3_1(x))
        x = self.dropout3(x)
        x = self.fc3_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,24:40]
        x = self.conv_block4_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc4_1(x))
        x = self.dropout4(x)
        x = self.fc4_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss

class Mfcc_SubFrebands_CNN(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Mfcc_SubFrebands_CNN, self).__init__()

        self.activation = activation
        
        self.conv_block1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block1_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block1_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block1_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block2_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block2_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block2_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block3_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block3_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block4_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block4_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block4_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_4 = ConvBlock(in_channels=256, out_channels=512)
                              
                             
        self.fc1_1 = nn.Linear(512, 512, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc2_1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc3_1 = nn.Linear(512, 512, bias=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc4_1 = nn.Linear(512, 512, bias=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4_2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1_1)
        init_layer(self.fc1_2)
        init_layer(self.fc2_1)
        init_layer(self.fc2_2)
        init_layer(self.fc3_1)
        init_layer(self.fc3_2)
        init_layer(self.fc4_1)
        init_layer(self.fc4_2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = x3
        x = x[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        temp = x.clone()
        count = 0.
        
        x = temp[:,:,:,0:16]
        x = self.conv_block1_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block1_4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc1_1(x))
        x = self.dropout1(x)
        x = self.fc1_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        
        x = temp[:,:,:,8:24]
        x = self.conv_block2_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2_1(x))
        x = self.dropout2(x)
        x = self.fc2_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,16:32]
        x = self.conv_block3_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc3_1(x))
        x = self.dropout3(x)
        x = self.fc3_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,24:40]
        x = self.conv_block4_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc4_1(x))
        x = self.dropout4(x)
        x = self.fc4_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss
    
class Cqt_SubFrebands_CNN(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Cqt_SubFrebands_CNN, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.CQT(sr=config.sample_rate, hop_length=config.hop_size, fmin=220, fmax=None, n_bins=64, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect')
        self.conv_block1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block1_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block1_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block1_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block2_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block2_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block2_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block3_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block3_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block4_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block4_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block4_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block5_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block5_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block5_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block5_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block6_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block6_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block6_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block6_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block7_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block7_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block7_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block7_4 = ConvBlock(in_channels=256, out_channels=512)
                              
                             
        self.fc1_1 = nn.Linear(512, 512, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc2_1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc3_1 = nn.Linear(512, 512, bias=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc4_1 = nn.Linear(512, 512, bias=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc5_1 = nn.Linear(512, 512, bias=True)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc5_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc6_1 = nn.Linear(512, 512, bias=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc6_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc7_1 = nn.Linear(512, 512, bias=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc7_2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1_1)
        init_layer(self.fc1_2)
        init_layer(self.fc2_1)
        init_layer(self.fc2_2)
        init_layer(self.fc3_1)
        init_layer(self.fc3_2)
        init_layer(self.fc4_1)
        init_layer(self.fc4_2)
        init_layer(self.fc5_1)
        init_layer(self.fc5_2)
        init_layer(self.fc6_1)
        init_layer(self.fc6_2)
        init_layer(self.fc7_1)
        init_layer(self.fc7_2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = input
        x = self.spec_layer(x1)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        temp = x.clone()
        count = 0.
        
        x = temp[:,:,:,0:16]
        x = self.conv_block1_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block1_4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc1_1(x))
        x = self.dropout1(x)
        x = self.fc1_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        
        x = temp[:,:,:,8:24]
        x = self.conv_block2_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2_1(x))
        x = self.dropout2(x)
        x = self.fc2_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,16:32]
        x = self.conv_block3_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc3_1(x))
        x = self.dropout3(x)
        x = self.fc3_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,24:40]
        x = self.conv_block4_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc4_1(x))
        x = self.dropout4(x)
        x = self.fc4_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,32:48]
        x = self.conv_block5_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block5_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block5_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc5_1(x))
        x = self.dropout5(x)
        x = self.fc5_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,40:56]
        x = self.conv_block6_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block6_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block6_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc6_1(x))
        x = self.dropout6(x)
        x = self.fc6_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,48:64]
        x = self.conv_block7_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block7_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block7_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block7_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc7_1(x))
        x = self.dropout7(x)
        x = self.fc7_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss

class Gamm_SubFrebands_CNN(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Gamm_SubFrebands_CNN, self).__init__()

        self.activation = activation
        self.conv_block1_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block1_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block1_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block1_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block2_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block2_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block2_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block3_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block3_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block3_4 = ConvBlock(in_channels=256, out_channels=512)
                              
        self.conv_block4_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block4_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block4_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block5_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block5_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block5_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block5_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block6_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block6_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block6_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block6_4 = ConvBlock(in_channels=256, out_channels=512)
        
        self.conv_block7_1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block7_2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block7_3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block7_4 = ConvBlock(in_channels=256, out_channels=512)
                              
                             
        self.fc1_1 = nn.Linear(512, 512, bias=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc2_1 = nn.Linear(512, 512, bias=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc3_1 = nn.Linear(512, 512, bias=True)
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc3_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc4_1 = nn.Linear(512, 512, bias=True)
        self.dropout4 = nn.Dropout(p=0.5)
        self.fc4_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc5_1 = nn.Linear(512, 512, bias=True)
        self.dropout5 = nn.Dropout(p=0.5)
        self.fc5_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc6_1 = nn.Linear(512, 512, bias=True)
        self.dropout6 = nn.Dropout(p=0.5)
        self.fc6_2 = nn.Linear(512, classes_num, bias=True)
        
        self.fc7_1 = nn.Linear(512, 512, bias=True)
        self.dropout7 = nn.Dropout(p=0.5)
        self.fc7_2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1_1)
        init_layer(self.fc1_2)
        init_layer(self.fc2_1)
        init_layer(self.fc2_2)
        init_layer(self.fc3_1)
        init_layer(self.fc3_2)
        init_layer(self.fc4_1)
        init_layer(self.fc4_2)
        init_layer(self.fc5_1)
        init_layer(self.fc5_2)
        init_layer(self.fc6_1)
        init_layer(self.fc6_2)
        init_layer(self.fc7_1)
        init_layer(self.fc7_2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = x2
        x = x[:,None,:,:]
        '''(batch_size, 1, times_steps, freq_bins)'''
        temp = x.clone()
        count = 0.
        
        x = temp[:,:,:,0:16]
        x = self.conv_block1_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block1_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block1_4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc1_1(x))
        x = self.dropout1(x)
        x = self.fc1_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        
        x = temp[:,:,:,8:24]
        x = self.conv_block2_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2_1(x))
        x = self.dropout2(x)
        x = self.fc2_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,16:32]
        x = self.conv_block3_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc3_1(x))
        x = self.dropout3(x)
        x = self.fc3_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,24:40]
        x = self.conv_block4_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block4_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc4_1(x))
        x = self.dropout4(x)
        x = self.fc4_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,32:48]
        x = self.conv_block5_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block5_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block5_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block5_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc5_1(x))
        x = self.dropout5(x)
        x = self.fc5_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,40:56]
        x = self.conv_block6_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block6_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block6_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block6_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc6_1(x))
        x = self.dropout6(x)
        x = self.fc6_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        x = temp[:,:,:,48:64]
        x = self.conv_block7_1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block7_2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block7_3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block7_4(x, pool_size=(2, 2), pool_type='avg')
        (x, _) = torch.max(x[:,:,:,0], dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc7_1(x))
        x = self.dropout7(x)
        x = self.fc7_2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x + co_output
        count += 1.
        outputs.append(x)
        
        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss
    
class Logmel_MultiFrames_CNN(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Logmel_MultiFrames_CNN, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = input
        x = self.spec_layer(x1)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')   
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') 
        x = torch.mean(x, dim=3)
        temp = x.clone()
        
        count = 0.
        x = temp[:,:,0]
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        for i in range(1,temp.shape[2],1):
            x = temp[:,:,i]
            x = F.relu_(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = x.view(x.shape[0], 1, x.shape[1])
            co_output = x + co_output
            count += 1.
            outputs.append(x)

        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss

class Cqt_MultiFrames_CNN(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Cqt_MultiFrames_CNN, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.CQT(sr=config.sample_rate, hop_length=config.hop_size, fmin=220, fmax=None, n_bins=64, bins_per_octave=12, norm=1, window='hann', center=True, pad_mode='reflect')
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = input
        x = self.spec_layer(x1)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')   
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') 
        x = torch.mean(x, dim=3)
        temp = x.clone()
        
        count = 0.
        x = temp[:,:,0]
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        for i in range(1,temp.shape[2],1):
            x = temp[:,:,i]
            x = F.relu_(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = x.view(x.shape[0], 1, x.shape[1])
            co_output = x + co_output
            count += 1.
            outputs.append(x)

        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss
    
class Gamm_MultiFrames_CNN(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Gamm_MultiFrames_CNN, self).__init__()

        self.activation = activation
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = x2
        x = x[:,None,:,:]
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')   
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') 
        x = torch.mean(x, dim=3)
        temp = x.clone()
        
        count = 0.
        x = temp[:,:,0]
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        for i in range(1,temp.shape[2],1):
            x = temp[:,:,i]
            x = F.relu_(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = x.view(x.shape[0], 1, x.shape[1])
            co_output = x + co_output
            count += 1.
            outputs.append(x)

        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss
    
class Mfcc_MultiFrames_CNN(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Mfcc_MultiFrames_CNN, self).__init__()

        self.activation = activation
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc1 = nn.Linear(512, 512, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, scale_num, total_frames)'''
        outputs = []
        
        x = x3
        x = x[:,None,:,:]
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')   
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg') 
        x = torch.mean(x, dim=3)
        temp = x.clone()
        
        count = 0.
        x = temp[:,:,0]
        x = F.relu_(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.view(x.shape[0], 1, x.shape[1])
        co_output = x
        count += 1.
        outputs.append(x)
        
        for i in range(1,temp.shape[2],1):
            x = temp[:,:,i]
            x = F.relu_(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            x = x.view(x.shape[0], 1, x.shape[1])
            co_output = x + co_output
            count += 1.
            outputs.append(x)

        outputs.append(co_output/count)
        outputs = torch.cat((outputs), 1)
        loss = torch.softmax(outputs,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(outputs, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(outputs)
            
        return output,loss
class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=99):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0.0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=None):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.attention_dropout = attention_dropout
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attetention张量
        """
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.attention_dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, v), p_attn


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=40, num_heads=4, output_dim=40, dropout=0.0, share_weight=False):
        super(MultiHeadAttention, self).__init__()

        self.share_weight = share_weight
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        if not share_weight:
            print('[MultiHead] No share weight.')
            self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
            self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        else:
            print('[MultiHead] Share weight.')
        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final_output = nn.Linear(model_dim, output_dim)
        self.linear_final_context = nn.Linear(model_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm_output = nn.LayerNorm(output_dim)
        self.layer_norm_context = nn.LayerNorm(output_dim)

    def forward(self, key, query, value, attn_mask=None):
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        value = self.linear_v(value)
        if not self.share_weight:
            key = self.linear_k(key)
            query = self.linear_q(query)
            # split by heads
        else:
            key = value
            query = value
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        # scale = (key.size(-1) // num_heads) ** -0.5
        scale = None
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        context += residual

        # final linear projection
        output = self.linear_final_output(context)
        context = self.linear_final_context(context)

        # add residual and norm layer
        output = self.layer_norm_output(output)
        context = self.layer_norm_context(context)

        output = F.relu(output)

        # dropout
        # output = self.dropout(output)

        return output, attention, context

    
class TCMHBlock(nn.Module):
    def __init__(self, c_in, c_out, s, max_len, output_dim):
        super(TCMHBlock, self).__init__()
        self.TConv = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(3, 1), stride=s, bias=False, padding=(1, 0)),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=(3, 1), stride=1, bias=False, padding=(1, 0)),
            nn.BatchNorm2d(c_out)
        )
        if s != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1), stride=s, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU()
            )
        else:
            self.shortcut = nn.Sequential()

        self.pe = PositionalEncoding(d_model=c_out, dropout=0.1, max_len=max_len)
        self.mh = MultiHeadAttention(model_dim=c_out, num_heads=4, output_dim=output_dim, dropout=0.1,
                                     share_weight=False)

    def forward(self, x):
        res = self.shortcut(x)  # (16,c_out,t',1)
        x = self.TConv(x)  # (16,c_out,t',1)
        x += res
        x = F.relu(x)

        x = torch.squeeze(x)  # (16,c_out,t')
        x = torch.transpose(x, 1, 2)  # (16,t',c_out)
        x = self.pe(x)  # (16,t',c_out)
        x, _, _ = self.mh(x, x, x)  # (16,t',c_out)
        x = torch.transpose(x, 1, 2)  # (16,c_out,t')
        x = torch.unsqueeze(x, 3)  # (16,c_out,t',1)

        return x

class Logmel_CnnMH(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Logmel_CnnMH, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=13)
        self.mh = MultiHeadAttention(model_dim=512, num_heads=8, output_dim=512, dropout=0.1,
                                     share_weight=False)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x1
        x = self.spec_layer(x)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        #if self.training:
            #x = self.spec_augmenter(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)         # (b, c, t)
        x = torch.transpose(x, 1, 2)      # (b, t, c)
        x = self.pe(x)               
        x, _, _ = self.mh(x, x, x)      
        x = torch.transpose(x, 1, 2)      # (b, c, t)
        
        x = torch.mean(x, dim=2)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Ensemble_CNN3(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Ensemble_CNN3, self).__init__()

        self.activation = activation
        self.model1 = Logmel_MultiFrebands_CNN(classes_num, activation, fixed=False)
        self.model2 = Logmel_SubFrebands_CNN(classes_num, activation, fixed=False)
        self.model3 = Logmel_MultiFrames_CNN(classes_num, activation, fixed=False)
        
        self.ite = 15000 #12000
        self.model_path = 'D:/Project/DCASE_test/checkpoints/liufeng/logmel_86frames_40melbins/TAU-urban-acoustic-scenes-2020-mobile-development/holdout_fold=1/'
        self.init_weights()

    def init_weights(self):
        model_path1 = self.model_path +'Logmel_MultiFrebands_CNN/'+str(self.ite)+'_iterations.pth'
        model_path2 = self.model_path +'Logmel_SubFrebands_CNN/'+str(self.ite)+'_iterations.pth'
        model_path3 = self.model_path +'Logmel_MultiFrames_CNN/'+str(self.ite)+'_iterations.pth'

        model_ch1 = torch.load(model_path1)
        model_ch2 = torch.load(model_path2)
        model_ch3 = torch.load(model_path3)
        
        self.model1.load_state_dict(model_ch1['model'])
        self.model2.load_state_dict(model_ch2['model'])
        self.model3.load_state_dict(model_ch3['model'])
       

    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.model1(input1,input2,input3,input4)
        x2,loss2 = self.model2(input1,input2,input3,input4)
        x3,loss3 = self.model3(input1,input2,input3,input4)
        
        x = x1[:,-1] + x2[:,-1] + x3[:,-1]
        loss = (loss1[:,-1]+loss2[:,-1]+loss3[:,-1])/3.
        return x,loss

class Ensemble_CNN5(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Ensemble_CNN5, self).__init__()

        self.activation = activation
        self.model1 = Logmel_MultiFrames_CNN(classes_num, activation, fixed=False)
        self.model2 = Cqt_MultiFrames_CNN(classes_num, activation, fixed=False)
        self.model3 = Gamm_MultiFrames_CNN(classes_num, activation, fixed=False)
        self.model4 = Mfcc_MultiFrames_CNN(classes_num, activation, fixed=False)
        
        self.ite = 15000 #12000
        self.model_path = 'D:/Project/DCASE_test/checkpoints/liufeng/logmel_86frames_40melbins/TAU-urban-acoustic-scenes-2020-mobile-development/holdout_fold=1/'
        self.init_weights()

    def init_weights(self):
        model_path1 = self.model_path +'Logmel_MultiFrames_CNN/'+str(self.ite)+'_iterations.pth'
        model_path2 = self.model_path +'Cqt_MultiFrames_CNN/'+str(self.ite)+'_iterations.pth'
        model_path3 = self.model_path +'Gamm_MultiFrames_CNN/'+str(self.ite)+'_iterations.pth'
        model_path4 = self.model_path +'Mfcc_MultiFrames_CNN/'+str(self.ite)+'_iterations.pth'

        model_ch1 = torch.load(model_path1)
        model_ch2 = torch.load(model_path2)
        model_ch3 = torch.load(model_path3)
        model_ch4 = torch.load(model_path4)
        
        self.model1.load_state_dict(model_ch1['model'])
        self.model2.load_state_dict(model_ch2['model'])
        self.model3.load_state_dict(model_ch3['model'])
        self.model4.load_state_dict(model_ch4['model'])
       

    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.model1(input1,input2,input3,input4)
        x2,loss2 = self.model2(input1,input2,input3,input4)
        x3,loss3 = self.model3(input1,input2,input3,input4)
        x4,loss4 = self.model4(input1,input2,input3,input4)
        
        x = x1[:,-1] + x2[:,-1] + x3[:,-1] + x4[:,-1]
        loss = (loss1[:,-1]+loss2[:,-1]+loss3[:,-1]+loss4[:,-1])/4.
        return x,loss

class Ensemble_CNN6(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Ensemble_CNN6, self).__init__()

        self.activation = activation
        self.model1 = Logmel_SubFrebands_CNN(classes_num, activation, fixed=False)
        self.model2 = Cqt_SubFrebands_CNN(classes_num, activation, fixed=False)
        self.model3 = Gamm_SubFrebands_CNN(classes_num, activation, fixed=False)
        self.model4 = Mfcc_SubFrebands_CNN(classes_num, activation, fixed=False)
        
        self.ite = 15000 #12000
        self.model_path = 'D:/Project/DCASE_test/checkpoints/liufeng/logmel_86frames_40melbins/TAU-urban-acoustic-scenes-2020-mobile-development/holdout_fold=1/'
        self.init_weights()

    def init_weights(self):
        model_path1 = self.model_path +'Logmel_SubFrebands_CNN/'+str(self.ite)+'_iterations.pth'
        model_path2 = self.model_path +'Cqt_SubFrebands_CNN/'+str(self.ite)+'_iterations.pth'
        model_path3 = self.model_path +'Gamm_SubFrebands_CNN/'+str(self.ite)+'_iterations.pth'
        model_path4 = self.model_path +'Mfcc_SubFrebands_CNN/'+str(self.ite)+'_iterations.pth'

        model_ch1 = torch.load(model_path1)
        model_ch2 = torch.load(model_path2)
        model_ch3 = torch.load(model_path3)
        model_ch4 = torch.load(model_path4)
        
        self.model1.load_state_dict(model_ch1['model'])
        self.model2.load_state_dict(model_ch2['model'])
        self.model3.load_state_dict(model_ch3['model'])
        self.model4.load_state_dict(model_ch4['model'])
       

    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.model1(input1,input2,input3,input4)
        x2,loss2 = self.model2(input1,input2,input3,input4)
        x3,loss3 = self.model3(input1,input2,input3,input4)
        x4,loss4 = self.model4(input1,input2,input3,input4)
        
        x = x1[:,-1] + x2[:,-1] + x3[:,-1] + x4[:,-1]
        loss = (loss1[:,-1]+loss2[:,-1]+loss3[:,-1]+loss4[:,-1])/4.
        return x,loss
    
class Ensemble_CNN4(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Ensemble_CNN4, self).__init__()

        self.activation = activation
        self.model1 = TFNet(classes_num, activation, fixed=False)
        self.model2 = Logmel_CnnMH(classes_num, activation, fixed=False)
        
        self.ite = 15000 #12000
        self.model_path = 'D:/Project/DCASE_test/checkpoints/liufeng/logmel_86frames_40melbins/TAU-urban-acoustic-scenes-2020-mobile-development/holdout_fold=1/'
        self.init_weights()

    def init_weights(self):
        model_path1 = self.model_path +'TFNet/'+str(self.ite)+'_iterations.pth'
        model_path2 = self.model_path +'Logmel_CnnMH/'+str(self.ite)+'_iterations.pth'

        model_ch1 = torch.load(model_path1)
        model_ch2 = torch.load(model_path2)
        
        self.model1.load_state_dict(model_ch1['model'])
        self.model2.load_state_dict(model_ch2['model'])

    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.model1(input1,input2,input3,input4)
        x2,loss2 = self.model2(input1,input2,input3,input4)
        
        x = x1 + x2
        loss = (loss1+loss2)/2.
        return x,loss
    
class Ensemble_Models(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Ensemble_Models, self).__init__()

        self.activation = activation
        self.model1 = Ensemble_CNN(classes_num, activation, fixed=False)
        self.model2 = Ensemble_CNN2(classes_num, activation, fixed=False)
        self.model3 = Ensemble_CNN3(classes_num, activation, fixed=False)
        self.model4 = Ensemble_CNN4(classes_num, activation, fixed=False)
        self.model5 = Ensemble_CNN5(classes_num, activation, fixed=False)
        self.model6 = Ensemble_CNN6(classes_num, activation, fixed=False)
        
        
    def forward(self, input1,input2,input3,input4):
        '''
        Input: (batch_size, total_frames)'''
        
        x1,loss1 = self.model1(input1,input2,input3,input4)
        x2,loss2 = self.model2(input1,input2,input3,input4)
        x3,loss3 = self.model3(input1,input2,input3,input4)
        x4,loss4 = self.model4(input1,input2,input3,input4)
        x5,loss5 = self.model5(input1,input2,input3,input4)
        x6,loss6 = self.model6(input1,input2,input3,input4)
        
#         x = x2/5. + x5/4. + x6/4.
#         loss = (loss2+loss5+loss6)/3.
        
#         x = x2 + x5 + x6
#         loss = (5.*loss2+4.*loss5+4.*loss6)/13.

        x = x1 + x2
        loss = (loss1+loss2)/2.

        return x,loss  

class FTB(nn.Module):

    def __init__(self, input_dim=40, in_channel=1, r_channel=16):

        super(FTB, self).__init__()
        self.in_channel = in_channel
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channel, r_channel, kernel_size=[1,1]),
                        nn.BatchNorm2d(r_channel),
                        nn.ReLU()
            )
        
        self.conv1d = nn.Sequential(
                        nn.Conv1d(r_channel*input_dim, in_channel, kernel_size=9,padding=4),
                        nn.BatchNorm1d(in_channel),
                        nn.ReLU()
            )
        self.freq_fc = nn.Linear(input_dim, input_dim, bias=False)

        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channel*2, in_channel, kernel_size=[1,1]),
                        nn.BatchNorm2d(in_channel),
                        nn.ReLU()
            )

    def forward(self, inputs):
        '''
        inputs should be [Batch, Ca, Dim, Time]
        '''
        # T-F attention 
        x = inputs.transpose(2,3)
        conv1_out = self.conv1(x)
        B, C, D, T= conv1_out.size()
        reshape1_out = torch.reshape(conv1_out,[B, C*D, T])
        conv1d_out = self.conv1d(reshape1_out)
        conv1d_out = torch.reshape(conv1d_out, [B, self.in_channel,1,T])
        
        # now is also [B,C,D,T]
        att_out = conv1d_out*x
        
        # tranpose to [B,C,T,D]
        att_out = torch.transpose(att_out, 2, 3)
        freqfc_out = self.freq_fc(att_out)
        att_out = torch.transpose(freqfc_out, 2, 3)

        cat_out = torch.cat([att_out, x], 1)
        outputs = self.conv2(cat_out)
        outputs = outputs.transpose(2,3)
        return outputs

class Logmel_FTB_Cnn(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Logmel_FTB_Cnn, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        self.FTB = FTB(input_dim=40, in_channel=1, r_channel=16)
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x1
        x = self.spec_layer(x)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        x = self.FTB(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

def getpad(x):
    x = x.transpose(2,3)
    a = (torch.arange(x.size(2)).float() * 2 /
         x.size(2) - 1.).unsqueeze(1).expand(-1, x.size(3)).unsqueeze(0).unsqueeze(0).expand(x.size(0), -1, -1, -1)
    return a.cuda().transpose(2,3)

class FConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_layer=False):
        super(FConvBlock, self).__init__()
        
        self.first_layer=first_layer
        if first_layer:
            self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels+1, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels+1, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        if not self.first_layer:
            x = torch.cat([x, getpad(x)], dim=1)
        x = F.relu_(self.bn1(self.conv1(x)))
        x = torch.cat([x, getpad(x)], dim=1)
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x

class Logmel_FCnn(nn.Module):
    
    def __init__(self, classes_num, activation, fixed=False):
        super(Logmel_FCnn, self).__init__()

        self.activation = activation
        self.spec_layer = Spec.MelSpectrogram(sr=config.sample_rate, n_fft=config.window_size, n_mels=config.mel_bins, hop_length=config.hop_size, window='hann', center=True, pad_mode='reflect', htk=False, fmin=0.0, fmax=None, norm=1, trainable_mel=False, trainable_STFT=False)
        #self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, freq_drop_width=5, freq_stripes_num=2)
        self.conv_block1 = FConvBlock(in_channels=1, out_channels=64, first_layer=True)
        self.conv_block2 = FConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = FConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = FConvBlock(in_channels=256, out_channels=512)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)
        init_layer(self.fc2)

    def forward(self, x1,x2,x3,x4):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = x1
        x = self.spec_layer(x)
        x = 10*torch.log10(x)
        x = torch.clamp(x, min=-100.)
        x = x.transpose(1,2)
        x = x[:,None,:,:]
        #if self.training:
            #x = self.spec_augmenter(x)
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(4, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(4, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        #x = F.dropout(x, p=0.2, training=self.training)
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = F.relu_(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc(x)
        loss = torch.softmax(x,-1)
        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)
            
        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)
        
        return output,loss

class Cnn_9layers_AvgPooling(nn.Module):

    def __init__(self, classes_num, activation):
        super(Cnn_9layers_AvgPooling, self).__init__()

        self.activation = activation

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):

        init_layer(self.fc)

    def forward(self, input):
        '''
        Input: (batch_size, times_steps, freq_bins)'''

        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''

        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)
        x = self.fc(x)

        if self.activation == 'logsoftmax':
            output = F.log_softmax(x, dim=-1)

        elif self.activation == 'sigmoid':
            output = torch.sigmoid(x)

        return output