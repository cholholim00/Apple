# BERT 모델 학습 환경 세팅

from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch

# 1. 모델 로드 (한국어 구어체 특화 모델 추천)
model = BertForSequenceClassification.from_pretrained("beomi/kcbert-base", num_labels=8)

# 2. 학습 파라미터 설정 (대회용 고성능 세팅)
training_args = TrainingArguments(
    output_dir='./results',          # 모델 저장 경로
    num_train_epochs=5,              # 10주 프로젝트니 5에포크 이상 권장
    per_device_train_batch_size=16,  # GPU 메모리에 맞춰 조절
    warmup_steps=500,                # 학습 초기 안정화
    weight_decay=0.01,               # 오버피팅 방지
    logging_dir='./logs',
    evaluation_strategy="epoch"      # 에포크마다 성능 검증
)

# 3. Trainer 실행 (사용자님이 준비한 Dataset 연결)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,     # 앞서 만든 DataLoader 결과물
    eval_dataset=val_dataset
)

trainer.train()