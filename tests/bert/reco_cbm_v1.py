# 필요한 라이브러리 임포트
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# 데이터 전처리 함수
def preprocess_text(text):
    if pd.isna(text) or text == 'Unknown':
        return ''
    return text

# KoBERT 모델과 토크나이저 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertModel.from_pretrained('skt/kobert-base-v1')

# 텍스트를 임베딩 벡터로 변환하는 함수
def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()[0]

# 데이터프레임에서 overview 임베딩 생성
def create_embeddings(df):
    embeddings = []
    for text in df['overview']:
        processed_text = preprocess_text(text)
        embedding = get_embedding(processed_text)
        embeddings.append(embedding)
    return np.array(embeddings)

# 추천 함수
def get_recommendations(contentid, df, embeddings, n_recommendations=5):
    # 입력된 contentid의 인덱스 찾기
    idx = df[df['contentid'] == contentid].index[0]
    
    # 코사인 유사도 계산
    sim_scores = cosine_similarity([embeddings[idx]], embeddings)
    sim_scores = sim_scores[0]
    
    # 유사도 점수와 인덱스를 페어로 만들어 정렬
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 자기 자신을 제외하고 상위 n개 추천
    sim_scores = sim_scores[1:n_recommendations+1]
    
    # 추천 결과 반환
    recommended_ids = [df.iloc[idx]['contentid'] for idx, _ in sim_scores]
    return recommended_ids

# 사용 예시
df = pd.read_csv('tests/bert/batch/batch_0.csv')
embeddings = create_embeddings(df)

# 특정 contentid에 대한 추천 받기
recommended_places = get_recommendations(2750144, df, embeddings)
print("추천된 장소:", recommended_places)