import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers.data.data_collator import DataCollatorForTokenClassification

# Dados fictícios de treino e teste
x_train = ["This is a sample input sentence 1", "Another sample input sentence 2", "Yet another sample sentence 3"]
y_train = [0, 1, 0]  # Rótulos fictícios para os exemplos de treino

x_test = ["Sample test sentence 1", "Another sample test sentence 2"]
y_test = [1, 0]  # Rótulos fictícios para os exemplos de teste

# Carregar o tokenizer e o modelo
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenizar os dados
def encode_data(texts, labels, tokenizer, max_len=512):
    input_ids = []
    attention_masks = []
    
    for text in texts:
        encoded_data = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids.append(encoded_data['input_ids'])
        attention_masks.append(encoded_data['attention_mask'])
    
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0), torch.tensor(labels)

train_texts = [" ".join([str(word) for word in x.split()]) for x in x_train]
test_texts = [" ".join([str(word) for word in x.split()]) for x in x_test]

train_inputs, train_masks, train_labels = encode_data(train_texts, y_train, tokenizer)
test_inputs, test_masks, test_labels = encode_data(test_texts, y_test, tokenizer)

# Definir o conjunto de dados
batch_size = 16

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# Definir o data collator personalizado
class MyDataCollator:
    def __call__(self, features):
        input_ids = torch.stack([f[0] for f in features])
        attention_masks = torch.stack([f[1] for f in features])
        labels = torch.tensor([f[2] for f in features])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'labels': labels
        }

# Treinar o modelo BERT
training_args = TrainingArguments(
    output_dir='./results',          
    num_train_epochs=3,              
    per_device_train_batch_size=16,  
    per_device_eval_batch_size=16,   
    warmup_steps=500,                
    weight_decay=0.01,               
    logging_dir='./logs',            
)

trainer = Trainer(
    model=model_bert,                         
    args=training_args,                  
    train_dataset=train_data,
    eval_dataset=test_data,
    data_collator=MyDataCollator()  # Usando o data collator personalizado
)

trainer.train()

# Salvar o modelo e o tokenizer
model_bert.save_pretrained("./model")
tokenizer.save_pretrained("./tokenizer")
