import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden
