# BERT Fine-tuning
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# 1. 데이터 로드
df = pd.read_csv("dataset/bridgeboard_train_v1.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['sentence'].tolist(), df['label'].tolist(), test_size=0.1, random_state=42
)

# 2. 토크나이저 및 모델 세팅 (KcBERT: 구어체 특화)
model_name = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class BridgeDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

train_dataset = BridgeDataset(train_encodings, train_labels)
val_dataset = BridgeDataset(val_encodings, val_labels)

# 3. 모델 정의 (8개 클래스)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=8)

# 4. 학습 설정 (대회 1등을 위한 하이퍼파라미터)
training_args = TrainingArguments(
    output_dir='models/bridgeboard_model',
    num_train_epochs=10,              # 5~10 에포크 추천
    per_device_train_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy="epoch",      # 매 에포크마다 검증
    save_total_limit=2,               # 용량 관리
    fp16=True if torch.cuda.is_available() else False # GPU 가속
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

# 5. 학습 시작
print("Starting Fine-tuning... 🚀")
trainer.train()
trainer.save_model("models/final_bridgeboard_model")
print("✅ Model Saved Successfully!")