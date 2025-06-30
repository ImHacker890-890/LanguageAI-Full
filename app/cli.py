import torch
from audio_utils import record_audio, play_audio
from models.seq2seq import Seq2Seq
from models.encoder import Encoder
from models.decoder import Decoder
from utils.tokenizer import text_to_sequence, sequence_to_text

class Translator:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = Encoder(10000, 256).to(self.device)
        self.decoder = Decoder(10000, 256).to(self.device)
        self.load_weights()

    def load_weights(self):
        """Загрузка весов модели"""
        try:
            self.encoder.load_state_dict(torch.load('models/weights/encoder.pth', map_location=self.device))
            self.decoder.load_state_dict(torch.load('models/weights/decoder.pth', map_location=self.device))
        except FileNotFoundError:
            raise Exception("Веса модели не найдены. Обучите модель сначала")

    def translate(self, text: str) -> str:
        """Полный цикл перевода"""
        # Токенизация
        seq = text_to_sequence(text)
        src = torch.tensor(seq).unsqueeze(1).to(self.device)
        
        # Перевод
        with torch.no_grad():
            hidden = self.encoder(src)
            output = self.decoder(src, hidden)
        
        # Детокенизация
        return sequence_to_text(output.argmax(dim=2).squeeze().tolist())

def main():
    translator = Translator()
    print("Голосовой переводчик (RU->EN). Нажмите Ctrl+C для выхода")
    
    while True:
        try:
            if (ru_text := record_audio()):
                en_text = translator.translate(ru_text)
                print(f"[Оригинал] {ru_text}")
                print(f"[Перевод] {en_text}")
                play_audio(en_text)
        except KeyboardInterrupt:
            print("\nЗавершение работы")
            break
        except Exception as e:
            print(f"[Ошибка] {e}")

if __name__ == "__main__":
    main()
