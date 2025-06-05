import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imdb = load_dataset("imdb")

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

imdb = imdb.map(tokenize_batch, batched=True)
imdb.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

train_loader = DataLoader(imdb['train'], batch_size=16, shuffle=True)

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, input_ids):
        embeds = self.embedding(input_ids)  
        pooled = embeds.mean(dim=1)         
        return self.fc(pooled).squeeze()

vocab_size = tokenizer.vocab_size
model = SentimentClassifier(vocab_size).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train():
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(train_loader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].float().to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()
    torch.save(model.state_dict(), "models/imdb_sentiment.pth")