import pandas as pd
import os
import openpyxl

def create_training_data():
    # 1. 업로드된 파일명 (복사해서 붙여넣으세요)
    train_file = "dataset/감성대화/Training_221115_add/원천데이터/감성대화말뭉치(최종데이터)_Training.xlsx"
    
    print("데이터 로딩 중... 📂")
    df = pd.read_excel(train_file)

    # 2. BridgeBoard용 8개 클래스 매핑 사전 (대회용 세팅)
    # AI Hub의 6개 대분류를 우리의 협업 8개 클래스로 매핑합니다.
    emotion_mapping = {
        '기쁨': 1,    # 동의/격려
        '당황': 3,    # 질문/확인
        '불안': 3,    # 질문/확인
        '상처': 4,    # 비판적 검토 (논리적 상처/반대)
        '분노': 5,    # 갈등/공격
        '슬픔': 7     # 기타/잡담
    }

    # 3. 데이터 추출 (사람문장 1, 2, 3 통합)
    print("문장 추출 및 라벨링 중... 🏷️")
    final_data = []
    
    for _, row in df.iterrows():
        label = emotion_mapping.get(row['감정_대분류'], 7) # 매핑 없으면 기타
        
        # 사람문장 1~3까지 모두 학습 데이터로 활용
        for col in ['사람문장1', '사람문장2', '사람문장3']:
            sentence = row[col]
            if pd.notna(sentence) and len(str(sentence)) > 2:
                final_data.append({
                    'sentence': str(sentence).strip(),
                    'label': label
                })

    # 4. 결과 저장
    result_df = pd.DataFrame(final_data)
    result_df.to_csv("dataset/bridgeboard_train_v1.csv", index=False, encoding='utf-8-sig')
    print(f"✅ 학습 데이터 생성 완료! 총 {len(result_df)}개의 문장이 준비되었습니다.")
    print("파일명: bridgeboard_train_v1.csv")

if __name__ == "__main__":
    create_training_data()