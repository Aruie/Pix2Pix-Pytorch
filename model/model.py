import torch
import torch.nn as nn


class Generator(nn.Module) :
    def __init__(self, input_channel, output_channel, is_unet = False) :
        super(Generator, self).__init__()
     
    def forward(self, x) :
        pass   
    

class Encoder(nn.Module) :
    def __init__(self) :
        super(Encoder, self).__init__()
        pass

    def forward(self, x) :
        pass

class Decoder(nn.Module) :
    def __init__(self) :
        super(Decoder, self).__init__()
    
        pass
    
    def forward(self, x) :
        pass

class Discriminator(nn.Module) :
    pass

if __name__=='__main__' :
    pass