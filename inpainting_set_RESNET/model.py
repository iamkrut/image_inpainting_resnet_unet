import torch
import torch.nn as nn
import torch.nn.functional as F

class resBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, downsample=False, upsample=False):
        super(resBlock, self).__init__()

        self.inchannel = inchannel
        self.outchannel = outchannel
        self.downsample = downsample
        self.upsample = upsample

        self.reflectpad1 = nn.ReflectionPad2d(1)

        if (self.downsample == True):
            self.down = nn.Sequential(nn.ReflectionPad2d(1),
                                      nn.Conv2d(inchannel,outchannel,kernel_size =3,stride=2,padding=0,bias=False),
                                      nn.BatchNorm2d(outchannel),
                                      nn.ReLU(),
                                      nn.ReflectionPad2d(1),
                                      nn.Conv2d(outchannel,outchannel,kernel_size =3,stride=1,padding=0,bias=False),
                                      nn.BatchNorm2d(outchannel))
            self.skipDown = nn.Conv2d(inchannel,outchannel,kernel_size =1,stride=2,padding=0,bias=False)


        if (self.downsample == False and self.upsample == False):
            self.standard = nn.Sequential(nn.ReflectionPad2d(1),
                            nn.Conv2d(inchannel,outchannel,kernel_size =3,stride=1,padding=0,bias=False),
                            nn.BatchNorm2d(outchannel),
                            nn.ReLU(),
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(outchannel,outchannel,kernel_size =3,stride=1,padding=0,bias=False),
                            nn.BatchNorm2d(outchannel))


        if (self.upsample == True):
            self.up = nn.Sequential(nn.ReflectionPad2d(1),
                                    nn.ConvTranspose2d(inchannel,outchannel,kernel_size =3,stride=2,padding=3,output_padding=1,bias=False),
                                    nn.BatchNorm2d(outchannel),
                                    nn.ReLU(),
                                    nn.ReflectionPad2d(1),
                                    nn.Conv2d(outchannel,outchannel,kernel_size =3,stride=1,padding=0,bias=False),
                                    nn.BatchNorm2d(outchannel))
            self.skipUp = nn.Conv2d(inchannel,outchannel,kernel_size =1,stride=1,padding=0,bias=False)






    def forward(self, x):

        if (self.downsample == True):
            x1 = self.down(x)
            x1 += self.skipDown(x)


        if (self.downsample == False and self.upsample == False):
            x1 = self.standard(x)
            x1 += x


        if (self.upsample == True):
            x1 = self.up(x)
            x2 = self.skipUp(x)
            x1 += F.interpolate(x2,scale_factor=2)


        return F.relu(x1)



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.beforeRes = nn.Sequential(nn.ReflectionPad2d(3),
                         nn.Conv2d(4,32,kernel_size =7,stride=1,padding=0,bias=False),
                         nn.ReLU(),
                         nn.ReflectionPad2d(2),
                         nn.Conv2d(32,64,kernel_size =5,stride=2,padding=0,bias=False),
                         nn.ReLU())

        self.block0 = resBlock(64,128,downsample=True,upsample=False)

        self.block1 = nn.ModuleList([resBlock(128,128,downsample=False,upsample=False ) for i in range(6)])

        self.block2 = resBlock(128,64,downsample=False,upsample=True)

        self.afterRes = nn.Sequential(nn.ReflectionPad2d(1),
                        nn.ConvTranspose2d(64,32,kernel_size =5,stride=2,padding=4,output_padding=1,bias=False),
                        nn.ReLU(),
                        nn.ReflectionPad2d(3),
                        nn.Conv2d(32,3,kernel_size =7,stride=1,padding=0,bias=True),
                        nn.Sigmoid())


    def forward(self, x):

        x1 = self.beforeRes(x)

        x1 = self.block0(x1)

        for i in range(6):
            x1 = self.block1[i](x1)

        x1 = self.block2(x1)

        x1 = self.afterRes(x1)

        return x1





