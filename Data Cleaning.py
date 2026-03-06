# 데이터 전처리 스크립트
import pandas as pd
from konlpy.tag import Kkma

kkma = Kkma()

def preprocess_bridgeboard(file_path):
    # 데이터 로드
    df = pd.read_csv(file_path)
    
    # 1. 필요한 컬럼만 추출 (문장과 감정 대분류)
    # 사람문장 1, 2, 3을 모두 합쳐서 데이터 양을 늘립니다.
    sentences = df['사람문장1'].tolist() + df['사람문장2'].tolist() + df['사람문장3'].tolist()
    labels_raw = df['감정_대분류'].tolist() * 3
    
    refined_data = []
    
    for text, label in zip(sentences, labels_raw):
        if pd.isna(text): continue
        
        # 2. Kkma를 활용한 전처리 (명사/동사 등 의미어 위주 추출 혹은 노이즈 제거)
        # 여기서는 모델 학습을 위해 문장 전체를 쓰되, 특수문자 등을 정제합니다.
        clean_text = text.strip() 
        
        # 3. 레이블 매핑 (사용자님의 8개 클래스에 맞춰 수정)
        mapping = {
            '기쁨': 1, # 동의/격려
            '분노': 5, # 갈등/공격
            '상처': 5,
            '불안': 3, # 질문/확인 (불안한 맥락의 재확인)
            '당황': 3,
            '슬픔': 7  # 기타
        }
        
        refined_data.append({
            'sentence': clean_text,
            'label': mapping.get(label, 7) # 매핑 없으면 기타
        })
        
    return pd.DataFrame(refined_data)

# 실행
train_df = preprocess_bridgeboard('Training_Validation.csv')
train_df.to_csv('bridgeboard_train1.csv', index=False)