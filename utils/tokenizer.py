from torchtext.data import get_tokenizer

tokenizer_ru = get_tokenizer('spacy', language='ru_core_news_sm')
tokenizer_en = get_tokenizer('spacy', language='en_core_web_sm')
