import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # todo
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        self.down1 = downStep(4, 64)
        self.down2 = downStep(64, 128)
        self.down3 = downStep(128, 256)
        self.down4 = downStep(256, 512)
        self.down5 = downStep(512, 1024)
        self.up1 = upStep(1024, 512)
        self.up2 = upStep(512, 256)
        self.up3 = upStep(256, 128)
        self.up4 = upStep(128, 64, withReLU = False)
        #self.outputConv = nn.Conv2d(64, n_classes, kernel_size = 1)
        self.maxpool2d = nn.MaxPool2d(kernel_size = 2, stride = 2)

    def forward(self, x):
        # todo
        #print(x.shape)
        x1 = self.down1(x)
        x2 = self.maxpool2d(x1)
        #print("x1 ", x1.shape)
        x3 = self.down2(x2)
        x4 = self.maxpool2d(x3)
        #print("x3", x3.shape)
        x5 = self.down3(x4)
        x6 = self.maxpool2d(x5)
        #print("x5", x5.shape)
        x7 = self.down4(x6)
        x8 = self.maxpool2d(x7)
        #print("x7", x7.shape)
        x9 = self.down5(x8)
        #print("x9", x9.shape)

        x = self.up1(x9, x7)
        #print("x10", x.shape)
        x = self.up2(x, x5)
        #print("x11", x.shape)
        x = self.up3(x, x3)
        #print("x12", x.shape)
        x = self.up4(x, x1)
        #print("x13", x.shape)
        #x = self.outputConv(x)
        #print(x.shape)
        return x

class downStep(nn.Module):
    def __init__(self, inC, outC):
        super(downStep, self).__init__()
        # todo
        self.convLayer1 = nn.Conv2d(inC, outC, kernel_size = 3, padding= 1)
        self.convLayer2 = nn.Conv2d(outC, outC, kernel_size = 3, padding= 1)
        #self.batchNorm = nn.BatchNorm2d(outC)
        self.relU = nn.ReLU(inplace=True)

    def forward(self, x):
        # todo
        x = self.convLayer1(x)
        x = self.relU(x)
        #x = self.batchNorm(x)
        x = self.convLayer2(x)
        x = self.relU(x)
        #x = self.batchNorm(x)
        return x

class upStep(nn.Module):
    def __init__(self, inC, outC, withReLU=True):
        super(upStep, self).__init__()
        # todo
        # Do not forget to concatenate with respective step in contracting path
        # Hint: Do not use ReLU in last convolutional set of up-path (128-64-64) for stability reasons!
        #self.up = nn.ConvTranspose2d(inC, inC//2, 2, stride=2)
        self.convLayer1 = nn.Conv2d(inC, outC, kernel_size = 3, padding= 1)
        self.convLayer2 = nn.Conv2d(outC, outC, kernel_size = 3, padding= 1)
        self.convLayer3 = nn.Conv2d(outC, 3, kernel_size = 3, padding= 1)
        self.deconvLayer1 = nn.ConvTranspose2d(inC, outC , kernel_size = 2, stride=2)
        self.sig = nn.Sigmoid()
        self.relU = nn.ReLU(inplace=True)
        self.withReLU = withReLU

    def forward(self, x, x_down):
        # todo
        x = self.deconvLayer1(x)

        # _, _, x_down_height, x_down_width = x_down.size()
        # diff_y = (x_down_height - x.shape[2:][0]) // 2
        # diff_x = (x_down_width - x.shape[2:][1]) // 2
        # crop = x_down[:, :, diff_y:(diff_y + x.shape[2:][0]), diff_x:(diff_x + x.shape[2:][1])]

        x = torch.cat([x, x_down], 1)
        x = self.convLayer1(x)
        if self.withReLU:
            x = self.relU(x)
            #x = self.batchNorm(x)
        x = self.convLayer2(x)
        if self.withReLU:
            x = self.relU(x)
            #x = self.batchNorm(x)
        else:
            x = self.convLayer3(x)
            x = self.sig(x)
        return x
