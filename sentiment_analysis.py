import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config import Config

def preprocess_text(text, tokenizer, max_len):
    # Pre-processamento do texto
    encoded_text = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return encoded_text

def analyze_sentiment(text, models, tokenizer):
    model_lstm, model_cnn, model_bert = models
    encoded_text = pad_sequences(
        [[tokenizer.get_word_index().get(word, 0) for word in text.split()]],
        maxlen=Config.MAX_LEN
    )

    # Predição LSTM
    lstm_pred = model_lstm.predict(encoded_text)[0][0]

    # Predição CNN
    cnn_pred = model_cnn.predict(encoded_text)[0][0]

    # Predição BERT
    inputs = preprocess_text(text, tokenizer, Config.MAX_LEN)
    outputs = model_bert(**inputs)
    bert_pred = torch.sigmoid(outputs.logits).detach().cpu().numpy()[0][0]

    # Média das predições
    avg_pred = np.mean([lstm_pred, cnn_pred, bert_pred])

    return "positive" if avg_pred > 0.5 else "negative"
