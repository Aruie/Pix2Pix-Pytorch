import torch
import torch.nn as nn


class Generator(nn.Module) :
    def __init__(self, is_unet = False) :
        super(Generator, self).__init__()

        self.is_unet = is_unet

        if self.is_unet == False :
            self.encoder = Encoder()
            self.decoder = Decoder()
        else :
            self.unet = UNET()

    def forward(self, x) :
        
        if self.is_unet == False :
            x = self.encoder(x)
            x = self.decoder(x)
        else :
            x = self.unet(x)
        
        return x



class UNET(nn.Module) : 
    def __init__(self) :
        super(UNET, self).__init__()

        self.encoder = nn.ModuleList()
        self.encoder.append( CK(1, 64, is_bn = False) )
        self.encoder.append( CK(64, 128) )
        self.encoder.append( CK(128, 256) )
        self.encoder.append( CK(256, 512) )
        self.encoder.append( CK(512, 512) )
        self.encoder.append( CK(512, 512) )
        self.encoder.append( CK(512, 512) )

        self.latent = CK(512, 512, is_bn = False)

        self.decoder = nn.ModuleList(
            [CDK(512, 512, is_upsample = True),
            CDK(1024, 512, is_upsample = True),
            CDK(1024, 512, is_upsample = True),
            CK(1024, 512, is_upsample = True),
            CK(1024, 256, is_upsample = True),
            CK(512, 128, is_upsample = True),
            CK(256, 64, is_upsample = True)]
        )
        self.out = CK(128, 3, is_upsample = True)
        self.act = nn.Tanh()
    
    def forward(self, x) :
        x1 = self.encoder[0](x)
        x2 = self.encoder[1](x1)
        x3 = self.encoder[2](x2)
        x4 = self.encoder[3](x3)
        x5 = self.encoder[4](x4)
        x6 = self.encoder[5](x5)
        x7 = self.encoder[6](x6)

        mid = self.latent(x7)

        x = self.decoder[0](mid)
        x = self.decoder[1](torch.cat([x, x7], dim=1) )
        x = self.decoder[2](torch.cat([x, x6], dim=1) )
        x = self.decoder[3](torch.cat([x, x5], dim=1) )
        x = self.decoder[4](torch.cat([x, x4], dim=1) )
        x = self.decoder[5](torch.cat([x, x3], dim=1) )
        x = self.decoder[6](torch.cat([x, x2], dim=1) )
        x = self.out(torch.cat([x, x1], dim=1) )
        
        x = self.act(x)
        return x

class CK(nn.Module) :
    def __init__(self, input_channel, output_channel, is_upsample = False, is_bn = True) :
        super(CK, self).__init__()
        self.is_bn = is_bn
        self.is_upsample = is_upsample

        if is_upsample == False :
            self.conv = nn.Conv2d(input_channel, output_channel, (4,4),stride=2, padding=1)
        else :
            self.conv = nn.ConvTranspose2d(input_channel, output_channel, (4,4), stride=2, padding=1)
        
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x) :
        x = self.conv(x)
        
        if self.is_bn == True :
            x = self.bn(x)
        x = self.relu(x)

        return x

class CDK(nn.Module) :
    def __init__(self, input_channel, output_channel, is_upsample = False, is_bn = True) :
        super(CDK, self).__init__()
        self.is_bn = is_bn
        self.is_upsample = is_upsample

        if self.is_upsample == False :
            self.conv = nn.Conv2d(input_channel, output_channel, (4,4),stride=2, padding=1)
        else :
            self.conv = nn.ConvTranspose2d(input_channel, output_channel, (4,4), stride=2, padding=1)
        
        self.bn = nn.BatchNorm2d(output_channel)
        self.dropout = nn.Dropout2d(0.5)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x) :
        x = self.conv(x)
        
        if self.is_bn == True :
            x = self.bn(x)
        x = self.dropout(x)
        x = self.relu(x)
        
        return x

class Encoder(nn.Module) :
    def __init__(self) :
        super(Encoder, self).__init__()

        self.layers = nn.Sequential(
            CK(1, 64, is_bn = False),
            CK(64, 128),
            CK(128, 256),
            CK(256, 512),
            CK(512,512),
            CK(512,512),
            CK(512,512)
        )

    def forward(self, x) :
        x = self.layers(x)
        return x


class Decoder(nn.Module) :
    def __init__(self) :
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            CDK(512, 512, is_upsample = True),
            CDK(512, 512, is_upsample = True),
            CDK(512, 512, is_upsample = True),
            CK(512, 512, is_upsample = True),
            CK(512, 256, is_upsample = True),
            CK(256, 128, is_upsample = True),
            CK(128, 64, is_upsample = True)
        )
        self.out = nn.ConvTransposed2d(64, 3, (4,4), strides = 2, padding=1)
    
    def forward(self, x) :
        x = self.layers(x)
        return x

class Discriminator(nn.Module) :
    def __init__(self) :
        super(Discriminator, self).__init__()

        self.layer1 = CK(3, 64, is_bn = False)        
        self.layer2 = CK(64, 128)        
        self.layer3 = CK(128, 256)        
        self.layer4 = CK(256, 512)        

        self.patch = nn.Conv2d(512, 1, (16,16))

        self.act = nn.Sigmoid()

    def forward(self, x) :
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.patch(x)
        x = self.act(x)
        batch = x.shape[0]
        x = x.view((batch, -1))
        x = torch.mean(x, dim=1)
    
        return x


if __name__=='__main__' :
    model = Generator(is_unet = True)
    
    x_input = torch.randn([1,3,512,512])
    discri = Discriminator()
    y = discri(x_input)
    print(y.shape)
    #x_input = torch.randn([1,1,512,512])
    #y = model(x_input)


    print(y.shape)

