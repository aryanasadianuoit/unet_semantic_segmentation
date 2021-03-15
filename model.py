import torch
from torch import nn
from torchsummary import summary
from torch.nn import functional as F


class Conv_Module(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Conv_Module, self).__init__()
        # we have set bias = False, since we use batchnorm after the convolution
        #Note that the original paper has been published before the introduction of batch norm
        #padding  = 1 , so this is a same convlution( for easier compatibleness with different input images
        self.conv_1 = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels,bias=False,padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_channels=out_channels, kernel_size=3, out_channels=out_channels,bias=False,padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)

    def forward(self,x):

        x = self.relu(self.bn_1(self.conv_1(x)))
        x = self.relu(self.bn_2(self.conv_2(x)))
        return x







class UNET(nn.Module):
    def __init__(self,in_channels=3,out_channels=1,dimensions=[64,128,256,512]):
        super(UNET,self).__init__()

        self.contarcting_list = nn.ModuleList()
        self.expansive_list = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(2,stride=2)

        #stacking the contracting part( left side of the architecture)
        for dimension in dimensions:
            self.contarcting_list.append(Conv_Module(in_channels,dimension))
            in_channels = dimension

        # stacking the Expansive part( Right side of the architecture)
        for dimension in reversed(dimensions):
            self.expansive_list.append(
                #dimension * 2 ===> we have concatenation through channels dimesnion
                nn.ConvTranspose2d(dimension * 2, dimension, kernel_size=2, stride=2))

            self.expansive_list.append(Conv_Module(in_channels=dimension*2,out_channels=dimension))



        #Bottom point ( lowest point of U-Net)

        self.bottom_part = Conv_Module(dimensions[-1],dimensions[-1]*2)


        # last 1-1 conv layer

        self.conv11 = nn.Conv2d(dimensions[0],out_channels=out_channels,kernel_size=1)



    def forward(self,x):

        # store the values from the contracting path which are needed to be concatenated in the Expansive path
        skip_connections =[]

        # travese the contratcing path
        for contract_module in self.contarcting_list:
            x = contract_module(x)
            skip_connections.append(x)
            x = self.maxpool(x)

        x = self.bottom_part(x)

        #traverse the expansive path

        #reverse the skip_connection list

        skip_connections = skip_connections[::-1]
        for idx in range(0,len(self.expansive_list),2):

            x = self.expansive_list[idx](x)
            skip_connection = skip_connections[idx//2]

            #in the paper they have used cropping for size mismatch in the time of concatentaion
            if x.shape != skip_connection.shape:
                #print("x.shape == >",x.shape)
                #print("skipp connection .shape == >",skip_connection.shape)

                # this is cropping solution that the paper has proposed, But by using this technique,
                #the output is not in the same spatial size of the input
                #skip_connection = skip_connection[::,::,0:x.shape[2], 0:x.shape[3]]


                #Better Solution, resize the input x to have the size of skip-sonnection
                if x.shape[2] > skip_connection.shape[2]:
                    skip_connection = F.interpolate(skip_connection, size=x.shape[2])
                if x.shape[2] < skip_connection.shape[2]:
                    x = F.interpolate(x, size=skip_connection.shape[2])

                    #todo handle asymetric inputs




                if x.shape[3] > skip_connection.shape[3]:
                    skip_connection = F.interpolate(skip_connection, size=x.shape[3])
                    #print("here 3")
                if x.shape[3] < skip_connection.shape[3]:
                    x = F.interpolate(x, size=skip_connection.shape[3])
                    #print("here 4")


                #print("x.shape after == >", x.shape)
                #print("skipp connection .shape == >", skip_connection.shape)


            concatenated_input = torch.cat((skip_connection,x),dim=1)
            x = self.expansive_list[idx+1](concatenated_input)



        x = self.conv11(x)


        return x





#test = UNET(in_channels=3,out_channels=3)

#virtual_input = torch.rand((1,3,572,572))

#out = test(virtual_input)
#print(out.shape)
''
#summary(test,input_size=(3,572 , 572),device="cpu")
#summary(test,input_size=(3,960 , 960),device="cpu")