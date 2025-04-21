import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Veri setini yükle
df = pd.read_csv("KAGGLE.csv")

# Etiketlerin benzersiz değerlerini kontrol edelim
#print("Unique labels in Outcome Variable:", df['Outcome Variable'].unique())


from transformers import BertTokenizer

"""# Tokenizer'ı yükleyelim
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")

# Tokenize fonksiyonunu yazalım
def tokenize_function(examples):
    return tokenizer(examples['Disease'], padding="max_length", truncation=True)

# Veri setini tokenize etme
df['tokenized'] = df["Disease"].apply(lambda x: tokenize_function({"Disease": x}))

# Tokenize edilmiş veriyi kontrol et
print(df.head())


#from transformers import BertForTokenClassification"""

"""# Modeli yükleyelim
model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")
from torch.utils.data import DataLoader"""
"""from torch.utils.data import DataLoader  # DataLoader'ı import et


# Tokenize edilmiş veriyi DataLoader ile yükleyelim
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Veriyi kontrol edelim
for batch in train_loader:
    print(batch)
    break  # İlk batch'i kontrol etmek için"""
    


from sklearn.preprocessing import LabelEncoder

# Etiketleri sayısal hale getirme
label_encoder = LabelEncoder()
df['encoded_labels'] = label_encoder.fit_transform(df['Outcome Variable'])

# Yeni etiketlerinizi kontrol edin
print("Encoded labels:", df['encoded_labels'].unique())

# İlk birkaç satırı kontrol edelim
print(df.head())



from transformers import BertTokenizer

# Tokenizer'ı yükleyelim
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")



    
from torch.utils.data import Dataset, DataLoader
import torch

# Custom Dataset sınıfını tanımlıyoruz
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, labels):
        self.data = data
        self.tokenizer = tokenizer
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx]  # Veri satırını alıyoruz
        label = self.labels.iloc[idx]  # Etiketi alıyoruz
        # Tokenize işlemi
        tokenized_input = self.tokenizer(text, truncation=True, padding='max_length', max_length=128)
        # Etiketi de ekliyoruz
        tokenized_input['labels'] = torch.tensor(label)
        return {key: torch.tensor(val) for key, val in tokenized_input.items()}

# Dataset'e yükleme
dataset = CustomDataset(df['Disease'], tokenizer, df['encoded_labels'])

# DataLoader'ı ayarlama
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# İlk batch'i kontrol edelim
print(next(iter(train_loader)))

