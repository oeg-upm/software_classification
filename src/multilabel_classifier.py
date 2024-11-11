import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# Step 1: Load and Preprocess the Data
# Load data
df = pd.read_csv("corpus_for_classifier.csv")

# Convert descriptions to strings and check class distribution
df['description'] = df['description'].astype(str)
print("Class Distribution:\n", df[['Library', 'Benchmark', 'Service', 'Workflow', 'Other']].sum())

# Split into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['description'].tolist(),
    df[['Library', 'Benchmark', 'Service', 'Workflow', 'Other']].values,
    test_size=0.3,
    random_state=42
)

# Initialize the DistilBERT tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Tokenize the text
train_encodings = tokenizer(list(map(str, train_texts)), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(list(map(str, test_texts)), truncation=True, padding=True, max_length=512)

# Step 2: Create a PyTorch Dataset
class MultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = MultiLabelDataset(train_encodings, train_labels)
test_dataset = MultiLabelDataset(test_encodings, test_labels)

# Step 3: Define the Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

# Step 4: Set Up Training Arguments and Trainer
training_args = TrainingArguments(
    output_dir='./results_classifier',
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs'
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Step 5: Train the Model
trainer.train()

# Step 6: Evaluate the Model
# Adjust threshold to see if it improves results
THRESHOLD = 0.3  # Experiment with different threshold values
predictions = trainer.predict(test_dataset)
preds = torch.sigmoid(torch.tensor(predictions.predictions)).numpy() > THRESHOLD  # Binarize predictions at adjusted threshold

# Calculate metrics
print(classification_report(test_labels, preds, target_names=['Library', 'Benchmark', 'Service', 'Workflow', 'Other'], zero_division=0))

model.save_pretrained('multilabel_software_classifier')
tokenizer.save_pretrained('multilabel_software_classifier')