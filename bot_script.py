import os
import time
from datetime import datetime
from instagrapi import Client
from dotenv import load_dotenv
import openai
import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification

# Carrega as variáveis de ambiente do arquivo .env.local
load_dotenv('.env.local')

# Acessa a variável de ambiente OPENAI_API_KEY
openai.api_key = os.getenv('OPENAI_API_KEY')

class InstagramBot:
    def __init__(self, username, password):
        self.username = username
        self.password = password
        self.client = Client()
        self.logged_in = False
        self.last_checked = datetime.now()
        self.message_history = {}

        # Carregar os modelos
        self.model_lstm = load_model('models/model_lstm.h5')
        self.model_cnn = load_model('models/model_cnn.h5')
        self.model_bert = BertForSequenceClassification.from_pretrained('models/model_bert')
        self.tokenizer = BertTokenizer.from_pretrained('models/tokenizer_bert')

    def sign_in(self):
        try:
            self.client.login(self.username, self.password)
            self.logged_in = True
            print(f"Logged in as {self.username}")
        except Exception as e:
            print(f"Failed to login: {e}")

    def sign_out(self):
        if self.logged_in:
            self.client.logout()
            self.logged_in = False
            print("Logged out")

    def send_message(self, recipient_id, message):
        try:
            self.client.direct_send(message, [recipient_id])
            print(f"Message sent to {recipient_id}")
        except Exception as e:
            print(f"Failed to send message to {recipient_id}: {e}")

    def check_direct_messages(self):
        try:
            threads = self.client.direct_threads()
            new_last_checked = datetime.now()
            for thread in threads:
                for item in thread.messages:
                    message_time = item.timestamp
                    if message_time > self.last_checked:
                        if item.user_id != self.client.user_id:
                            user_info = self.client.user_info(item.user_id)
                            username = user_info.username if user_info else "Unknown"
                            
                            if username == 'Instagram_username':
                                print(f"{username}: {item.text}")

                                if username not in self.message_history or item.text != self.message_history[username]:
                                    sentiment = self.analyze_sentiment(item.text)
                                    response = self.get_openai_chat_completion(item.text, sentiment)
                                    if response:
                                        self.send_message(item.user_id, response)
                                        self.message_history[username] = item.text

            self.last_checked = new_last_checked
        except Exception as e:
            print(f"Failed to fetch direct messages: {e}")

    def listen_for_messages(self):
        try:
            while True:
                self.check_direct_messages()
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopping...")

    def get_openai_chat_completion(self, query, sentiment):
        try:
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"The sentiment of the user is {sentiment}."},
                    {"role": "user", "content": query}
                ]
            )
            return completion['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Erro ao processar a requisição da OpenAI: {e}")
            return None

    def analyze_sentiment(self, text):
        encoded_text = pad_sequences([imdb.get_word_index()[word] for word in text.split() if word in imdb.get_word_index()], maxlen=maxlen)
        
        # Predição LSTM
        lstm_pred = self.model_lstm.predict(encoded_text)[0][0]

        # Predição CNN
        cnn_pred = self.model_cnn.predict(encoded_text)[0][0]

        # Predição BERT
        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        bert_pred = self.model_bert(**inputs).logits.detach().cpu().numpy()[0][0]

        # Média das predições
        avg_pred = np.mean([lstm_pred, cnn_pred, bert_pred])

        return "positive" if avg_pred > 0.5 else "negative"

if __name__ == "__main__":
    bot = InstagramBot('Instagram_username', 'Instagram_password')
    bot.sign_in()
    recipient_username = 'Instagram_username'
    message = 'Olá Instagram_username! Esta é uma mensagem automatizada.'
    user_id = bot.client.user_id_from_username(recipient_username)
    bot.send_message(user_id, message)
    bot.listen_for_messages()
