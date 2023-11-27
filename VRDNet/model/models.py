import torch
import torch.nn as nn
import torch.nn.functional as F
from .deconv import FastDeconv
from .deform import DeformConv2d
import numpy as np
import time



class YCrCb(nn.Module):
    def __init__(self):
        super(YCrCb, self).__init__()
        self.y_weight = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).cuda()
        self.cb_weight = torch.tensor([-0.169, -0.331, 0.500]).view(1, 3, 1, 1).cuda()
        self.cr_weight = torch.tensor([0.500, -0.419, -0.081]).view(1, 3, 1, 1).cuda()

    def forward(self, img):
        y = F.conv2d(img, self.y_weight)
        cb = F.conv2d(img, self.cb_weight) + 0.5
        cr = F.conv2d(img, self.cr_weight) + 0.5
        #ycrcb = torch.cat([y, cr, cb], dim=1)
        #return ycrcb, y, torch.cat([cr, cb], dim=1)
        return torch.cat([cr, cb], dim=1)

class fusion(nn.Module):
    def __init__(self, in_features, out_features):
        super(fusion, self).__init__()
        self.fusion = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, out_features, 7, padding=0, bias=True),
            nn.Tanh()
        )

    def forward(self, input):
        out = self.fusion(input)
        return out

class HSV(nn.Module):
    def __init__(self):
        super(HSV, self).__init__()

    def forward(self, input):
        # Assuming input is in range [0, 1]
        r, g, b = input.split(1, dim=1)

        max_c, _ = input.max(dim=1, keepdim=True)
        min_c, _ = input.min(dim=1, keepdim=True)
        diff = max_c - min_c

        h = torch.where(max_c == min_c, torch.zeros_like(max_c),
                        torch.where(max_c == r, 60.0 * (g - b) / diff + 360.0,
                        torch.where(max_c == g, 60.0 * (b - r) / diff + 120.0,
                        60.0 * (r - g) / diff + 240.0)))
        h = h % 360

        s = torch.where(max_c == 0, torch.zeros_like(max_c), diff / max_c)
        v = max_c

        # Stack h, s, v channels
        out = torch.cat([v, s], dim=1)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, out_features, 3, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(in_features),
                        nn.PReLU(),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(out_features, out_features, 3, stride=1, padding=0, bias=True),
                        nn.BatchNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class PALayer(nn.Module):
    def __init__(self, nc, number):
        super(PALayer, self).__init__()
        self.conv = nn.Conv2d(nc,number,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(number)
        self.at = nn.Sequential(
            nn.Conv2d(number, 1, 1, stride=1, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(1, 1, 1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.at(y)
        return x * y

class CALayer(nn.Module):
    def __init__(self, nc, number):
        super(CALayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(nc)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.at = nn.Sequential(
            nn.Conv2d(nc, number, 1, stride=1, padding=0, bias=False),
            nn.PReLU(),
            nn.Conv2d(number, nc, 1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv(x)
        y = self.avg_pool(y)
        y = self.at(y)
        return x * y

class DehazeBlock(nn.Module):
    def __init__(self, nc, number=4):
        super(DehazeBlock, self).__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(nc)
        self.act1 = nn.PReLU()
        #self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(nc, number)
        self.palayer = PALayer(nc, number)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.act1(self.conv1(res))
        res = res + x
        #res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res
'''
class enhance(nn.Module):
    def __init__(self):
        super(enhance, self).__init__()
        #fusion1
        self.fusion1 = fusion(in_features=512, out_features=256)
        #self.fusion2 = fusion(in_features=4, out_features=64)

        #RGB2HSV
        self.hsv = HSV()

        #RGB2YCrCb
        self.ycrcb = YCrCb()

        #下采样输出1
        #self.rb1_1 = ResidualBlock(in_features=64, out_features=64)
        self.rb1_2 = ResidualBlock(in_features=2, out_features=64)
        self.rb1_3 = ResidualBlock(in_features=2, out_features=64)

        self.down1_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        
        self.down1_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        
        #下采样输出2
        self.rb2_1 = ResidualBlock(in_features=128, out_features=128)
        self.rb2_2 = ResidualBlock(in_features=128, out_features=128)
        self.rb2_3 = ResidualBlock(in_features=128, out_features=128)
        self.down2_1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        
        self.down2_2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )

        #输出3
        self.rb3_1 = ResidualBlock(in_features=256, out_features=256)
        self.rb3_2 = ResidualBlock(in_features=256, out_features=256)
        self.rb3_3 = ResidualBlock(in_features=256, out_features=256)

        #注意力块
        self.db1 = DehazeBlock(nc=64, number=4)
        self.db2 = DehazeBlock(nc=64, number=4)
        self.db3 = DehazeBlock(nc=64, number=4)
        #self.db4 = DehazeBlock(nc=256, number=4)
        #self.db5 = DehazeBlock(nc=256, number=4)
        #self.db6 = DehazeBlock(nc=256, number=4)


        """
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

        
        self.dc2 = nn.Sequential(
            DeformConv2d(256, 256),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        #解码器
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.rb4_1 = ResidualBlock(in_features=128, out_features=128)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.rb4_2 = ResidualBlock(in_features=64, out_features=64)
        """

    def forward(self, x):
        #颜色空间转化
        x1 = self.hsv(x)
        x2 = self.ycrcb(x)

        #特征融合
        x1 = self.rb1_2
        x3 = torch.cat([x1, x2], 1)
        x3 = self.fusion2(x3)
        x3 = self.db1(x3)
        x3 = self.db2(x3)
        x3 = self.db3(x3)
        #out1 = self.dc1(x3)
        out1 = self.conv(x3)

        #下采样1
        #x4 = self.conv(out1)
        #x4 = self.rb1_1(x4)
        #x4 = self.down1_1(x4)
        #x2 = self.rb1_2(x2)
        #x2 = self.down1_2(x2)

        #下采样2
        #x4 = self.rb2_1(x4)
        #x4 = self.down2_1(x4)
        #x2 = self.rb2_2(x2)
        #x2 = self.down2_2(x2)

        #输出
        #x4 = self.rb3_1(x4)
        #out = x4
        #x1 = self.dc1(x1)
        #x2 = self.rb3_2(x2)
        #x2 = self.dc2(x2)

        #融合输出1
        #x3 = torch.cat([x1, x2], 1)
        #x3 = self.fusion1(x3)
        #out = x3

        return out1
'''
class DehazeBlock(nn.Module):
    def __init__(self, nc, number=4):
        super(DehazeBlock, self).__init__()
        self.conv1 = nn.Conv2d(nc, nc, 3, 1, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(nc)
        self.act1 = nn.PReLU()
        #self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(nc, number)
        self.palayer = PALayer(nc, number)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.act1(self.conv1(res))
        res = res + x
        #res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class enhance(nn.Module):
    def __init__(self):
        super(enhance, self).__init__()
        #fusion1
        self.fusion1 = fusion(in_features=512, out_features=256)

        #RGB2HSV
        self.hsv = HSV()

        #RGB2YCrCb
        self.ycrcb = YCrCb()

        #下采样输出1
        self.rb1_1 = ResidualBlock(in_features=64, out_features=64)
        self.rb1_2 = ResidualBlock(in_features=64, out_features=64)
        #self.rb1_3 = ResidualBlock(in_features=3, out_features=64)
        self.down1_1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.down1_2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

        #下采样输出2
        self.rb2_1 = ResidualBlock(in_features=128, out_features=128)
        self.rb2_2 = ResidualBlock(in_features=128, out_features=128)
        #self.rb2_3 = ResidualBlock(in_features=128, out_features=128)
        self.down2_1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.down2_2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )

        #输出3
        self.rb3_1 = ResidualBlock(in_features=256, out_features=256)
        self.rb3_2 = ResidualBlock(in_features=256, out_features=256)
        #self.rb3_3 = ResidualBlock(in_features=256, out_features=256)

        #注意力块
        self.db1 = DehazeBlock(nc=256, number=4)
        self.db2 = DehazeBlock(nc=256, number=4)
        self.db3 = DehazeBlock(nc=256, number=4)
        #self.db4 = DehazeBlock(nc=256, number=4)
        #self.db5 = DehazeBlock(nc=256, number=4)
        #self.db6 = DehazeBlock(nc=256, number=4)

        """
        #解码器
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.rb4_1 = ResidualBlock(in_features=128, out_features=128)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.rb4_2 = ResidualBlock(in_features=64, out_features=64)
        """
        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

    def forward(self, x):
        #颜色空间转化
        x1 = self.hsv(x)
        x1 = self.conv1(x1)
        x2 = self.ycrcb(x)
        x2 = self.conv2(x2)

        #下采样1
        x1 = self.rb1_1(x1)
        x1 = self.down1_1(x1)
        x2 = self.rb1_2(x2)
        x2 = self.down1_2(x2)

        #下采样2
        x1 = self.rb2_1(x1)
        x1 = self.down2_1(x1)
        x2 = self.rb2_2(x2)
        x2 = self.down2_2(x2)

        #输出
        x1 = self.rb3_1(x1)
        x2 = self.rb3_2(x2)
        x3 = torch.cat([x1, x2], 1)
        x3 = self.fusion1(x3)
        x3 = self.db1(x3)
        x3 = self.db2(x3)
        x3 = self.db3(x3)
        out = x3

        return out
class VRDNet(nn.Module):
    def __init__(self):
        super(VRDNet, self).__init__()
        #融合输出2
        self.enhance = enhance()
        self.fusion2 = fusion(in_features=512, out_features=256)

        #下采样1
        self.rb1_3 = ResidualBlock(in_features=64, out_features=64)
        self.down1_3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )

        #下采样2
        self.rb2_3 = ResidualBlock(in_features=128, out_features=128)
        self.down2_3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )

        #输出
        self.rb3_3 = ResidualBlock(in_features=256, out_features=256)

        #注意力块
        self.db1 = DehazeBlock(nc=256, number=4)
        self.db2 = DehazeBlock(nc=256, number=4)
        self.db3 = DehazeBlock(nc=256, number=4)
        self.db4 = DehazeBlock(nc=256, number=4)
        self.db5 = DehazeBlock(nc=256, number=4)
        self.db6 = DehazeBlock(nc=256, number=4)

        #可变卷积
        self.dc3 = nn.Sequential(
            DeformConv2d(256, 256),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )
        self.dc4 = nn.Sequential(
            DeformConv2d(256, 256),
            nn.BatchNorm2d(256),
            nn.PReLU()
        )

        #上采样
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.rb4_1 = ResidualBlock(in_features=128, out_features=128)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.rb4_2 = ResidualBlock(in_features=64, out_features=64)
        self.fusion3 = fusion(in_features=64, out_features=3)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

    def forward(self, x):
        x1 = self.enhance(x)
        x2 = self.conv(x)
        x2 = self.rb1_3(x2)
        x2 = self.down1_3(x2)
        x2 = self.rb2_3(x2)
        x2 = self.down2_3(x2)
        x2 = self.rb3_3(x2)

        x3 = torch.cat([x1, x2], 1)
        x3 = self.fusion2(x3)

        x4 = self.db1(x3)
        x4 = self.db2(x4)
        x4 = self.db3(x4)
        x4 = self.db4(x4)
        x4 = self.db5(x4)
        x4 = self.db6(x4)

        x5 = self.up1(x4)
        x5 = self.rb4_1(x5)
        x5 = self.up2(x5)
        x5 = self.rb4_2(x5)
        out = self.fusion3(x5)

        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.BatchNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.BatchNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.BatchNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return torch.sigmoid(F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1))
        
if __name__ == '__main__':
    net = VRDNet().cuda()
    input_tensor = torch.Tensor(np.random.random((1,3,1500,1000))).cuda()
    start = time.time()
    out = net(input_tensor)
    end = time.time()
    print('Process Time: %f'%(end-start))
    print(input_tensor.shape) 

