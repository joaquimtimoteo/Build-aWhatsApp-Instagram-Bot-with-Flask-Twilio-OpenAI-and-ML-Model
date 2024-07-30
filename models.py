import torch
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertForSequenceClassification
from config import Config

class ModelLoader:
    def __init__(self):
        self.model_lstm = load_model(Config.LSTM_MODEL_PATH)
        self.model_cnn = load_model(Config.CNN_MODEL_PATH)
        self.model_bert = BertForSequenceClassification.from_pretrained(Config.BERT_MODEL_PATH)
        self.tokenizer = BertTokenizer.from_pretrained(Config.TOKENIZER_PATH)

    def get_models(self):
        return self.model_lstm, self.model_cnn, self.model_bert, self.tokenizer
