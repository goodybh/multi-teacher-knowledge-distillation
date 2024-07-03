import warnings
import os
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import datasets

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy.*")
warnings.filterwarnings("ignore", message=".*enable_nested_tensor is True, but self.use_nested_tensor is False.*")

# Suppress Hugging Face symlink warning
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Define transformer-based student model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(512, embed_dim)  # Assuming max sentence length of 512
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.dropout = nn.Dropout(dropout)
        self.qa_outputs = nn.Linear(embed_dim, 2)
    
    def forward(self, input_ids, attention_mask):
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).repeat(input_ids.size(0), 1).to(input_ids.device)
        x = self.embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(self.transformer_encoder(x.transpose(0, 1), src_key_padding_mask=~attention_mask.bool()).transpose(0, 1))
        logits = self.qa_outputs(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

# Parameters
embed_dim = 256
num_heads = 8
num_layers = 4
hidden_dim = 512
dropout = 0.1
learning_rate = 5e-5
weight_decay = 0.01
batch_size = 16

# Load pre-trained BERT model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
teacher = BertForQuestionAnswering.from_pretrained('bert-base-uncased').to(device)
student = TransformerModel(bert_tokenizer.vocab_size, embed_dim, num_heads, num_layers, hidden_dim, dropout).to(device)

# Load SQuAD dataset
dataset = datasets.load_dataset('squad')

# Preprocess the dataset
def preprocess_data(data):
    contexts = []
    questions = []
    answers = []
    for item in data:
        context = item['context']
        question = item['question']
        answer = item['answers']['text'][0]  # Use the first answer provided
        contexts.append(context)
        questions.append(question)
        answers.append(answer)
    return contexts, questions, answers

train_contexts, train_questions, train_answers = preprocess_data(dataset['train'])
valid_contexts, valid_questions, valid_answers = preprocess_data(dataset['validation'])

# Custom Dataset using BERT tokenizer
class SquadDataset(Dataset):
    def __init__(self, contexts, questions, answers, tokenizer, max_length=512):
        self.contexts = contexts
        self.questions = questions
        self.answers = answers
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.questions[idx],
            self.contexts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        start_idx = self.contexts[idx].find(self.answers[idx])
        end_idx = start_idx + len(self.answers[idx])
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor(start_idx, dtype=torch.long),
            'end_positions': torch.tensor(end_idx, dtype=torch.long)
        }

# Create dataset
train_dataset = SquadDataset(train_contexts, train_questions, train_answers, bert_tokenizer)
valid_dataset = SquadDataset(valid_contexts, valid_questions, valid_answers, bert_tokenizer)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# Fine-tune the teacher model on the downstream task
def fine_tune_teacher(model, train_loader, valid_loader, epochs=3):
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'start_positions': batch['start_positions'].to(device),
                'end_positions': batch['end_positions'].to(device)
            }
            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Teacher Model - Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(train_loader)}')
        evaluate_teacher(model, valid_loader)

# Evaluate function for teacher model
def evaluate_teacher(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'start_positions': batch['start_positions'].to(device),
                'end_positions': batch['end_positions'].to(device)
            }
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}')

# Evaluate function for student model
def evaluate_student(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            start_logits, end_logits = model(inputs, attention_mask)
            loss = (criterion(start_logits, start_positions) + criterion(end_logits, end_positions)) / 2
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}')

# Generate soft labels from the fine-tuned BERT model
def get_soft_labels(model, dataloader):
    model.eval()
    soft_labels = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device)
            }
            outputs = model(**inputs)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits
            soft_labels.append((start_logits, end_logits))
    start_logits, end_logits = zip(*soft_labels)
    return torch.cat(start_logits), torch.cat(end_logits)

# Combined loss function
def combined_loss(start_logits, end_logits, soft_start_logits, soft_end_logits, start_positions, end_positions, alpha=0.5):
    kl_loss = (nn.functional.kl_div(torch.log_softmax(start_logits, dim=1), soft_start_logits, reduction='batchmean') +
               nn.functional.kl_div(torch.log_softmax(end_logits, dim=1), soft_end_logits, reduction='batchmean')) / 2
    ce_loss = (nn.functional.cross_entropy(start_logits, start_positions) +
               nn.functional.cross_entropy(end_logits, end_positions)) / 2
    return alpha * kl_loss + (1 - alpha) * ce_loss

# Training loop for student model with soft and hard labels
def train_student(student, dataloader, soft_start_logits, soft_end_logits, valid_loader, epochs=5, alpha=0.5):
    student.train()
    optimizer = AdamW(student.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            optimizer.zero_grad()
            start_logits, end_logits = student(inputs, attention_mask)
            start_soft_label_batch = soft_start_logits[i * batch_size:(i + 1) * batch_size].to(device)
            end_soft_label_batch = soft_end_logits[i * batch_size:(i + 1) * batch_size].to(device)
            loss = combined_loss(start_logits, end_logits, start_soft_label_batch, end_soft_label_batch, start_positions, end_positions, alpha)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(dataloader)}')
        evaluate_student(student, valid_loader)

# Fine-tune and evaluate the teacher model
fine_tune_teacher(teacher, train_loader, valid_loader)

# Generate soft labels
soft_start_logits, soft_end_logits = get_soft_labels(teacher, train_loader)

# Train and evaluate the student model
train_student(student, train_loader, soft_start_logits, soft_end_logits, valid_loader)
