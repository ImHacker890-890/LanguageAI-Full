from models.seq2seq import Seq2Seq
from utils.dataloader import TranslationDataset

dataset = TranslationDataset(src_texts, trg_texts)
model = Seq2Seq(encoder, decoder)
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for src, trg in dataloader:
        optimizer.zero_grad()
        output = model(src, trg)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
