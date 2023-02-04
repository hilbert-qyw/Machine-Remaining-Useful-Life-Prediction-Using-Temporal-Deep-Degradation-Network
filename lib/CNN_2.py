import torch
from torch import tensor
import torch.nn as nn
from torch.nn.functional import dropout

def ConvBNReLU(in_channels,out_channels,kernel_size,stride=1,padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(inplace=True),
    )

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=5, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out


class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class CNN_Depthwise_Encoder(nn.Module):
    def __init__(self,in_places, places=36, output_features=64 ):
        super(CNN_Depthwise_Encoder,self).__init__()

        self.bottleneck_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,groups=in_places,kernel_size=(3,1),stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places,groups=places,kernel_size=(3,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),     

            nn.Conv2d(in_channels=places, out_channels=places,groups=places,kernel_size=(3,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),    

        )

        self.bottleneck_2 = nn.Sequential(
            nn.Conv2d(in_channels=places,out_channels=places,kernel_size=(5,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places,out_channels=places,kernel_size=(5,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(5,1), stride=2, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
        )

        self.cbam_1 = CBAM(channel=places)
        self.cbam_2 = CBAM(channel=places)
        
        self.con1x1 = nn.Conv2d(places, int(output_features/16), 1, bias=False)

        self.pooling = nn.AdaptiveAvgPool2d((16,1))
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, h, c, t = x.shape
        x = x.reshape(b*h, c, t, 1)
        #print(x.shape)

        out = self.bottleneck_1(x)
        out = self.cbam_1(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.bottleneck_2(out)
        out = self.cbam_2(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.con1x1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.pooling(out)

        out = out.reshape(b,h,-1)
        out = self.dropout(out)
        #print(out.shape)
        return out

class CNN_Depthwise_Encoder_1(nn.Module):
    def __init__(self,in_places, places=36, output_features=64 ):
        super(CNN_Depthwise_Encoder_1,self).__init__()

        self.bottleneck_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(5,1),stride=1, bias=False),
            # nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.05),


            nn.Conv2d(in_channels=places,out_channels=places,kernel_size=(5,1), stride=1, bias=False),
            # nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.05),  

            nn.Conv2d(in_channels=places,out_channels=places,kernel_size=(5,1), stride=2, bias=False),
            # nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.05),

        )

        self.cbam_1 = CBAM(channel=places)
        
        self.con1x1 = nn.Conv2d(places, int(output_features/16), 1, bias=False)

        self.pooling = nn.AdaptiveAvgPool2d((16,1))
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, h, c, t = x.shape
        x = x.reshape(b*h, c, t, 1)
        #print(x.shape)

        out = self.bottleneck_1(x)
        out = self.cbam_1(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.con1x1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.pooling(out)

        out = out.reshape(b,h,-1)
        out = self.dropout(out)
        #print(out.shape)
        return out

class CNN_Encoder(nn.Module):
    def __init__(self,in_places, places, ):
        super(CNN_Encoder,self).__init__()

        self.bottleneck_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(3,1),stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(3,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),     

        )

        self.bottleneck_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(5,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(5,1), stride=2, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
        )

        self.cbam_1 = CBAM(channel=places)
        self.cbam_2 = CBAM(channel=places)
        
        self.con1x1 = nn.Conv2d(places, int(places/8), 1, bias=False)

        self.pooling = nn.AdaptiveAvgPool2d((8,1))
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, h, c, t = x.shape
        x = x.reshape(b*h, c, t, 1)
        #print(x.shape)

        out = self.bottleneck_1(x)
        out = self.cbam_1(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.bottleneck_2(x)
        out = self.cbam_2(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.con1x1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.pooling(out)

        out = out.reshape(b,h,-1)
        out = self.dropout(out)
        #print(out.shape)
        return out



class CNN_Encoder_1(nn.Module):
    def __init__(self,in_places, places, ):
        super(CNN_Encoder_1,self).__init__()

        self.bottleneck_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(3,1),stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(3,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),     

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(3,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),     

        )

        self.bottleneck_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(5,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(5,1), stride=2, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(5,1), stride=2, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
        )

        self.cbam_1 = CBAM(channel=places)
        self.cbam_2 = CBAM(channel=places)
        
        self.con1x1 = nn.Conv2d(places, int(places/4), 1, bias=False)

        #self.pooling = nn.AdaptiveAvgPool2d((8,1))
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, h, c, t = x.shape
        x = x.reshape(b*h, c, t, 1)
        #print(x.shape)

        out = self.bottleneck_1(x)
        out = self.cbam_1(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.bottleneck_2(x)
        out = self.cbam_2(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.con1x1(out)
        out = self.relu(out)
        #print(out.shape)
        #out = self.pooling(out)

        out = out.reshape(b,h,-1)
        out = self.dropout(out)
        #print(out.shape)
        return out

class CNN_Encoder_2(nn.Module):
    def __init__(self,in_places, places,):
        super(CNN_Encoder_2,self).__init__()

        self.bottleneck_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(3,1),stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(3,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),     

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(3,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),     

        )

        self.bottleneck_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(5,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(5,1), stride=2, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(5,1), stride=2, bias=False),
            #nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
        )

        self.cbam_1 = CBAM(channel=places)
        self.cbam_2 = CBAM(channel=places)
        
        self.con1x1 = nn.Conv2d(places, int(places/4), 1, bias=False)

        self.pooling = nn.AdaptiveAvgPool2d((8,1))
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, h, c, t = x.shape
        x = x.reshape(b*h, c, t, 1)
        #print(x.shape)

        out = self.bottleneck_1(x)
        out = self.cbam_1(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.bottleneck_2(x)
        out = self.cbam_2(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.con1x1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.pooling(out)

        out = out.reshape(b,h,-1)
        out = self.dropout(out)
        #print(out.shape)
        return out

class CNN_Encoder_3(nn.Module):
    def __init__(self,in_places, places, output_features=64):
        super(CNN_Encoder_3,self).__init__()

        self.bottleneck_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(3,1),stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(3,1), stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.02, inplace=True),    

        )

        self.bottleneck_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(5,1), stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.AvgPool2d((2,1)),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(5,1), stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.AvgPool2d((2,1)),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.cbam_1 = CBAM(channel=places)
        self.cbam_2 = CBAM(channel=places)
        
        self.con1x1 = nn.Conv2d(places, int(output_features/8), 1, bias=False)

        self.pooling = nn.AdaptiveAvgPool2d((8,1))
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, h, c, t = x.shape
        x = x.reshape(b*h, c, t, 1)
        #print(x.shape)

        out = self.bottleneck_1(x)
        out = self.cbam_1(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.bottleneck_2(x)
        out = self.cbam_2(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.con1x1(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.pooling(out)

        out = out.reshape(b,h,-1)
        out = self.dropout(out)
        #print(out.shape)
        return out

class CNN_Encoder_4(nn.Module):
    def __init__(self,in_places, places, output_features=64):
        super(CNN_Encoder_4,self).__init__()

        self.bottleneck_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(3,1),stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(3,1), stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.02, inplace=True),    

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(3,1), stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.LeakyReLU(0.02, inplace=True),    

        )

        self.bottleneck_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(3,1), stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.AvgPool2d((2,1)),
            nn.LeakyReLU(0.02, inplace=True),

            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=(5,1), stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.AvgPool2d((2,1)),
            nn.LeakyReLU(0.02, inplace=True),
        )

        self.cbam_1 = CBAM(channel=places)
        self.cbam_2 = CBAM(channel=places)
        
        self.con1x1 = nn.Conv2d(places, int(output_features/16), 1, bias=False)

        self.pooling = nn.AdaptiveAvgPool2d((16,1))
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, h, c, t = x.shape
        x = x.reshape(b*h, c, t, 1)
        #print(x.shape)

        out = self.bottleneck_1(x)
        out = self.cbam_1(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.bottleneck_2(x)
        out = self.cbam_2(out)
        out = self.relu(out)
        #print(out.shape)

        out = self.con1x1(out)
        out = self.relu(out)
        #sprint(out.shape)
        out = self.pooling(out)

        out = out.reshape(b,h,-1)
        out = self.dropout(out)
        #print(out.shape)
        return out
    
class CNN_Encoder_5(nn.Module):
    def __init__(self,in_places, places=8, output_features=64):
        super(CNN_Encoder_5,self).__init__()

        self.bottleneck_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=(3,1),stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.LeakyReLU(0.02, inplace=True),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(in_channels=places, out_channels=places*2, kernel_size=(3,1), stride=1, bias=False),
            #nn.BatchNorm2d(places),
            nn.LeakyReLU(0.02, inplace=True), 
            nn.MaxPool2d((2, 1)),  
            nn.Dropout(0.2),   

        )

        self.cbam_1 = CBAM(channel=places*2)

        
        self.con1x1 = nn.Conv2d(places*2, int(output_features/8), 1, bias=False)

        self.pooling = nn.AdaptiveAvgPool2d((8, 1))
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, h, c, t = x.shape
        x = x.reshape(b*h, c, t, 1)
        #print(x.shape)

        out = self.bottleneck_1(x)
        out = self.cbam_1(out)
        out = self.relu(out)
        print(out.shape)

        #out = self.bottleneck_2(x)
        #out = self.cbam_2(out)
        #out = self.relu(out)
        #print(out.shape)

        out = self.con1x1(out)
        out = self.relu(out)
        print(out.shape)
        out = self.pooling(out)
        print(out.shape)

        out = out.reshape(b,h,-1)
        out = self.dropout(out)
        #print(out.shape)
        return out

class Conv1dNet(nn.Module):
    def __init__(self, kernkel_size=[3,4,4,3,4], strides=[1,2,2,2,2], ) -> None:
        super().__init__()
        layers = []
        for ker, stride in zip(kernkel_size, strides):
            #layers.append(Conv1dBlock(kernel_size=ker, strides=stride, ))
            pass
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)





if __name__ == '__main__':
    
    input = torch.randn((1, 10, 3, 55))

    model = CNN_Encoder_5(3, 16, 64)

    print(model(input).shape)

