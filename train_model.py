from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch
import pandas as pd
import numpy as np

# 1. Load Dataset
df = pd.read_csv("KAGGLE.csv")  # Load your CSV file here

# 2. Convert labels to numerical values
label_encoder = LabelEncoder()
df['encoded_labels'] = label_encoder.fit_transform(df['Outcome Variable'])
num_labels = len(label_encoder.classes_)
print(f"Number of unique classes: {num_labels}")

# 3. Split data into train and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['Disease'], df['encoded_labels'], test_size=0.2, random_state=42
)

# 4. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# 5. Define dataset class for sequence classification
class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Return as a dictionary with labels
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 6. Create datasets and dataloaders
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 7. Load the model for sequence classification
model = AutoModelForSequenceClassification.from_pretrained(
    "dmis-lab/biobert-v1.1", 
    num_labels=num_labels
)

# 8. Set device (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Using device: {device}")

# 9. Set optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

# 10. Training loop
epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print training loss
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()
            
            predictions = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (predictions == batch['labels']).sum().item()
            total_predictions += batch['labels'].size(0)
    
    val_accuracy = correct_predictions / total_predictions
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

# 11. Save the model
model.save_pretrained("./biobert_disease_classifier")
tokenizer.save_pretrained("./biobert_disease_classifier")
print("Model saved successfully!")

# 12. Save label encoder mapping for future predictions
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
print("Label encoder mapping saved!")

# Example of how to make a prediction with the trained model
def predict(text, model, tokenizer, label_encoder):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # Get model prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get predicted class
    predicted_class_idx = torch.argmax(outputs.logits, dim=1).item()
    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]
    
    return predicted_class

# Uncomment to test prediction on a sample
# sample_text = "Example disease description"
# prediction = predict(sample_text, model, tokenizer, label_encoder)
# print(f"Predicted class: {prediction}")