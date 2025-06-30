from models.encoder import Encoder
from models.decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # ... (код из предыдущего примера)
        return outputs
