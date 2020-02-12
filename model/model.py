import torch
import torch.nn as nn




class Generator(nn.Module) :
    def __init__(self) :
        super(Generator, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder(patch_size = 70)



class Encoder(nn.Module) :
    pass

class Decoder(nn.Module) :
    pass


class Discriminator(nn.Module) :
    pass










if __name__=='__main__' :
    pass
    