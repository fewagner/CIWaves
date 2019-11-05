import torch.nn as nn
import torch.nn.functional as F
import torch


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 9, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 9, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
         x = self.conv(x)
         return x        

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x
        
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(            
            nn.MaxPool1d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x
    
class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, 7, padding=1, stride=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)    
        )
        self.conv = double_conv(in_ch, out_ch)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        #here a padding error might appear because pytorch does not know same-padding
        diff = x2.size()[2] - x1.size()[2]

        x1 = F.pad(x1, (diff // 2, diff - diff//2)) #only last dim gets padded
        
        x = torch.cat([x2, x1], dim = 1) #
        x = self.conv(x)
        return x        

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, label_len):
        super(outconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, 50, 9, padding=3), #padding was choosen such that output fits label
            nn.BatchNorm1d(50),
            nn.ReLU(inplace=True),
            nn.Conv1d(50, out_ch, 1, padding=1)
        )
        self.label_len = label_len

    def forward(self, x):
        
        #here a padding error might appear because pytorch does not know same-padding
        diff = self.label_len - x.size()[2]
        x = F.pad(x, (diff//2, diff-diff//2))
        
        x = self.conv(x)
        return x
    

class UNet_featurek(nn.Module):
    def __init__(self, device, n_channels = 1, n_classes = 1, label_len = 10000):
        super(UNet_featurek, self).__init__()
        self.inc = inconv(n_channels, 128)
        self.down1 = down(128, 256)
        self.down2 = down(256, 512)
        self.down3 = down(512, 1024)
        self.down4 = down(1024, 2047)
        self.up1 = up(2048, 1024)
        self.up2 = up(1024, 512)
        self.up3 = up(512, 256)
        self.up4 = up(256, 128)
        self.outc = outconv(128, n_classes, label_len)
        self.device = device

    def forward(self, x0):
        
        x = x0[:,1:].to(self.device)
        k_values = x0[:,0]
        
        #print(k_values)
        
        # resize here the data first such that it matches the CNN layers
        x = x.view(len(x), -1, len(x[0]))
        
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        k = torch.ones(x5.size()[0], 1, x5.size()[2]).to(self.device)
        
        for sample,k_val in zip(k,k_values):
            sample[:,:] = k_val
            
        #print(k)
        
        features = torch.cat((k,x5), dim=1)
        
        x = self.up1(features, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x.view(len(x), -1) # if n_classes > 1 this should be adapted  