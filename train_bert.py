import os
import asyncio
from pymongo import MongoClient
from dotenv import load_dotenv
from instagram_bot import InstagramBot

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurações do MongoDB a partir das variáveis de ambiente
MONGO_URI = os.getenv('MONGO_URI')
client = MongoClient(MONGO_URI)
db = client['ChatAnalyticsDB']  # Nome do banco de dados
collection = db['Instagram_messages']  # Nome da coleção

async def save_message_to_db(message_data):
    try:
        # Inserir uma nova mensagem no banco de dados
        result = collection.insert_one(message_data)
        print(f"Mensagem salva com o id: {result.inserted_id}")
    except Exception as e:
        print(f"Erro ao salvar mensagem no banco de dados: {e}")

async def main():
    bot = InstagramBot(username='your_username', password='your_password')
    
    try:
        await bot.sign_in()  # Corrigido para aguardar a função assíncrona
    except Exception as e:
        print(f"Erro ao fazer login: {e}")
        return

    # Opcional: Enviar uma mensagem inicial para um usuário específico
    recipient_username = 'your__instagram_username'
    message = 'Hi instagram_username! This a automatized Mensage.'
    
    try:
        user_id = await bot.get_user_id(recipient_username)  # Usar função assíncrona para obter user_id
        if user_id:
            await bot.send_message(user_id, message)
    except Exception as e:
        print(f"Erro ao enviar mensagem: {e}")

    async def on_message_received(message):
        try:
            # Preparar os dados da mensagem para salvar no MongoDB
            message_data = {
                'user_id': message.get('user_id'),
                'username': message.get('username'),
                'content': message.get('content'),
                'timestamp': message.get('timestamp')
            }
            await save_message_to_db(message_data)
        except Exception as e:
            print(f"Erro ao processar mensagem recebida: {e}")

    try:
        # Entrar em um loop contínuo para verificar novas mensagens e respondê-las automaticamente
        await bot.listen_for_messages(on_message_received)
    except Exception as e:
        print(f"Erro ao escutar mensagens: {e}")

if __name__ == "__main__":
    asyncio.run(main())
