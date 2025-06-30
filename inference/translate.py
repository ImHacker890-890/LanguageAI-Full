from models.seq2seq import Seq2Seq

model = Seq2Seq(encoder, decoder)
model.load_state_dict(torch.load('model.pth'))

def translate(text):
    tokens = tokenize(text)
    output = model(tokens)
    return detokenize(output)
