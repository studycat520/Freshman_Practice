import torch
from torchsummary import summary

class BasicBlock(torch.nn.Module):
    def __init__(self,In_channel,Out_channel,Downsample=False):
        super(BasicBlock, self).__init__()
        self.stride = 1
        if Downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv2d(In_channel, Out_channel, 3, self.stride, padding=1),
            torch.nn.BatchNorm2d(Out_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(Out_channel, Out_channel, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(Out_channel),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv2d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual


class ResNet18(torch.nn.Module):
    def __init__(self, classes=10):
        super(ResNet18, self).__init__()
        self.features = torch.nn.Sequential(
            #conv1
            torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),

            #conv2_x
            torch.nn.MaxPool2d(3,2,1),
            BasicBlock(64, 64, False),
            BasicBlock(64, 64, False),

            # conv3_x
            BasicBlock(64, 128, True),
            BasicBlock(128, 128, False),

            # conv4_x
            BasicBlock(128, 256, True),
            BasicBlock(256, 256, False),

            # conv5_x
            BasicBlock(256, 512, True),
            BasicBlock(512, 512, False),

            torch.nn.AdaptiveAvgPool2d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(512,classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,512)
        x = self.classifer(x)
        return x

if __name__ == '__main__':


    x = torch.randn(size=(1,3,224,224))

    # x = torch.randn(size=(1,64,224))
    # model = Bottlrneck(64,64,256,True)
    model = ResNet18()


    output = model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    print(model)
    summary(model=model, input_size=(3, 224,224), device='cpu')


