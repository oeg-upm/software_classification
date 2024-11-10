# Step 1: Install required packages
# pip install transformers torch scikit-learn

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertForSequenceClassification
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Step 2: Define categories
categories = ["Library", "Service", "Workflow", "Benchmark", "Other"]
num_labels = len(categories)

# Step 3: Define a sample dataset (replace with your own data)
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

# Example dataset
texts = [
    "This is a library for data processing.",
    "We offer a benchmarking service.",
    "This workflow simplifies the process.",
    "This service is used for benchmarking."
]
labels = [
    [1, 0, 0, 0, 0],  # Library
    [0, 1, 0, 1, 0],  # Service, Benchmark
    [0, 0, 1, 0, 0],  # Workflow
    [0, 1, 0, 1, 0]   # Service, Benchmark
]

# Step 4: Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 128

# Step 5: Create dataset and dataloaders
dataset = CustomDataset(texts, labels, tokenizer, max_len)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Step 6: Define model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=num_labels,
    problem_type="multi_label_classification"
)
model.train()

# Step 7: Define optimizer and loss
optimizer = AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.BCEWithLogitsLoss()

# Step 8: Training loop (for simplicity, one epoch)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(1):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        print(f"Loss: {loss.item()}")

# Step 9: Evaluation (on the same dataset for simplicity)
model.eval()
preds, true_labels = [], []

with torch.no_grad():
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        preds.extend(torch.sigmoid(outputs.logits).cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# Thresholding and metrics
threshold = 0.5
preds = np.array(preds) > threshold
accuracy = accuracy_score(np.array(true_labels), preds)
f1 = f1_score(np.array(true_labels), preds, average='micro')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")