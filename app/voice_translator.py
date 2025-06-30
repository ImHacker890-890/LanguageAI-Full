import speech_recognition as sr
import pyttsx3
import torch
from models.seq2seq import Seq2Seq
from utils.tokenizer import tokenize, detokenize

# --- Инициализация модели ---
encoder = Encoder(input_size=10000, hidden_size=256)
decoder = Decoder(output_size=10000, hidden_size=256)
encoder.load_state_dict(torch.load("models/weights/encoder.pth"))
decoder.load_state_dict(torch.load("models/weights/decoder.pth"))
model = Seq2Seq(encoder, decoder)

# --- Голосовой ввод ---
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Говорите...")
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio, language="ru-RU")
            return text
        except:
            return ""

# --- Синтез речи ---
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# --- Перевод ---
def translate(text):
    tokens = tokenize(text)  # Твой токенизатор
    input_tensor = torch.tensor(tokens).unsqueeze(1)
    with torch.no_grad():
        output = model(input_tensor)
    return detokenize(output)  # Твой детокенизатор

# --- Запуск ---
if __name__ == "__main__":
    while True:
        ru_text = listen()
        if ru_text:
            en_text = translate(ru_text)
            print(f"Перевод: {en_text}")
            speak(en_text)
