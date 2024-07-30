from flask import Flask, request, jsonify
from twilio.twiml.messaging_response import MessagingResponse
import logging
import os
import openai
from dotenv import load_dotenv
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pymongo import MongoClient
from datetime import datetime, timezone

# Carrega as variáveis de ambiente do arquivo .env.local
load_dotenv('.env.local')

# Configura a chave da API do OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configura a conexão com o MongoDB
mongo_uri = os.getenv('MONGO_URI')
client = MongoClient(mongo_uri)
db = client['ChatAnalyticsDB']
collection = db['whatsapp_messages']

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Carregar os modelos LSTM, CNN e BERT uma única vez
try:
    model_lstm = load_model('models/model_lstm.h5')
    model_cnn = load_model('models/model_cnn.h5')
    model_bert = BertForSequenceClassification.from_pretrained('models/model_bert')
    tokenizer = BertTokenizer.from_pretrained('models/tokenizer_bert')
    logging.info("Modelos carregados com sucesso.")
except Exception as e:
    logging.error(f"Erro ao carregar os modelos: {e}")
    raise

def analyze_sentiment(text):
    try:
        # Pré-processar o texto para os modelos LSTM e CNN
        max_len = 128
        encoded_text = pad_sequences([tokenizer.encode(text, add_special_tokens=False)], maxlen=max_len)

        # Predição LSTM
        lstm_pred = model_lstm.predict(encoded_text)[0][0]

        # Predição CNN
        cnn_pred = model_cnn.predict(encoded_text)[0][0]

        # Predição BERT
        inputs = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        with torch.no_grad():
            outputs = model_bert(**inputs)
        bert_pred = outputs.logits.detach().cpu().numpy()[0][0]

        return float(lstm_pred), float(cnn_pred), float(bert_pred)
    except Exception as e:
        logging.error(f"Erro ao analisar o sentimento: {e}")
        return None, None, None

def get_openai_chat_completion(query, sentiment_data):
    try:
        lstm_pred, cnn_pred, bert_pred = sentiment_data

        # Determinar o sentimento médio com base nas predições dos modelos
        avg_pred = np.mean([lstm_pred, cnn_pred, bert_pred])
        sentiment = "positive" if avg_pred > 0.5 else "negative"
        
        # Enviar os resultados individuais dos modelos para a OpenAI
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Model predictions: LSTM={lstm_pred:.2f}, CNN={cnn_pred:.2f}, BERT={bert_pred:.2f}. The averaged sentiment is {sentiment}."},
                {"role": "user", "content": query}
            ]
        )
        return completion['choices'][0]['message']['content'].strip()
    except Exception as e:
        logging.error(f"Erro ao processar a requisição da OpenAI: {e}")
        return "Desculpe, não consegui processar sua requisição no momento."

@app.route('/bot', methods=['POST'])
def bot():
    try:
        incoming_msg = request.values.get('Body', '').strip()
        if not incoming_msg:
            logging.warning("Mensagem vazia recebida.")
            return jsonify({"status": "error", "message": "Mensagem vazia recebida."}), 400

        resp = MessagingResponse()
        msg = resp.message()

        logging.info(f"Mensagem recebida: {incoming_msg}")

        # Analisar o sentimento da mensagem recebida
        sentiment_data = analyze_sentiment(incoming_msg)
        if None in sentiment_data:
            return jsonify({"status": "error", "message": "Erro ao analisar o sentimento."}), 500

        logging.info(f"Sentimentos detectados: LSTM={sentiment_data[0]:.2f}, CNN={sentiment_data[1]:.2f}, BERT={sentiment_data[2]:.2f}")

        # Obter a resposta da OpenAI com base no sentimento e nas predições dos modelos
        openai_response = get_openai_chat_completion(incoming_msg, sentiment_data)
        
        # Enviar a resposta de volta ao usuário no WhatsApp
        msg.body(openai_response)
        
        logging.info(f"Resposta enviada: {openai_response}")

        # Armazenar a conversa no MongoDB
        conversation = {
            "incoming_msg": incoming_msg,
            "sentiment_data": {
                "LSTM": sentiment_data[0],
                "CNN": sentiment_data[1],
                "BERT": sentiment_data[2]
            },
            "response": openai_response,
            "timestamp": datetime.now(timezone.utc)
        }
        collection.insert_one(conversation)

        return str(resp)
    except Exception as e:
        logging.error(f"Ocorreu um erro: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(debug=True)
    except Exception as e:
        logging.error(f"Erro ao iniciar o servidor: {e}")
        raise
