import speech_recognition as sr
import pyttsx3
from typing import Optional

def record_audio(timeout: int = 3) -> Optional[str]:
    """Захват аудио с микрофона -> текст (русский)"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("[Говорите сейчас]")
        try:
            audio = r.listen(source, timeout=timeout)
            text = r.recognize_google(audio, language="ru-RU")
            return text
        except sr.WaitTimeoutError:
            print("[Таймаут]")
            return None
        except Exception as e:
            print(f"[Ошибка распознавания] {e}")
            return None

def play_audio(text: str, language: str = 'en') -> None:
    """Синтез речи через TTS"""
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('voice', language)
    engine.say(text)
    engine.runAndWait()
