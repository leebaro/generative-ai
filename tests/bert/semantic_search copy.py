# unset PYTHONPATH
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pickle
import os
import pandas as pd

class SemanticSearch:
    def __init__(self, model_name="jhgan/ko-sroberta-multitask"):
        """
        Args:
            model_name (str): 사용할 사전학습 모델 이름
        """
        self.model_name = model_name
        self.embedder = SentenceTransformer(model_name)
        self.corpus = []
        self.corpus_embeddings = None
        
    def add_corpus(self, corpus):
        """코퍼스 추가 및 임베딩 생성"""
        self.corpus = corpus
        self.corpus_embeddings = self.embedder.encode(corpus, convert_to_tensor=True)
        
    def search(self, query, top_k=5):
        """
        의미적 유사도 검색 수행
        Args:
            query (str): 검색할 문장
            top_k (int): 반환할 상위 결과 수
        Returns:
            list: (문장, 유사도 점수) 튜플의 리스트
        """
        query_embedding = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(query_embedding, self.corpus_embeddings)[0]
        cos_scores = cos_scores.cpu()
        
        top_k = min(len(self.corpus), top_k)
        top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]
        
        return [(self.corpus[idx].strip(), float(cos_scores[idx])) 
                for idx in top_results[0:top_k]]
    
    def save_model(self, path):
        """모델 상태 저장"""
        state = {
            'corpus': self.corpus,
            'corpus_embeddings': self.corpus_embeddings,
            'model_name': self.model_name
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(state, f)
            
    @classmethod
    def load_model(cls, path):
        """저장된 모델 상태 불러오기"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        searcher = cls(model_name=state['model_name'])
        searcher.corpus = state['corpus']
        searcher.corpus_embeddings = state['corpus_embeddings']
        return searcher 
    
    
def main():
    # CSV 파일 읽기
    df = pd.read_csv('tests/bert/batch/batch_0.csv')

    # overview가 Unknown이 아닌 행만 필터링
    df = df[df['overview'] != 'Unknown']

    # corpus를 overview 컬럼으로 지정
    corpus = df['overview'].tolist()

    # contentid도 함께 저장
    contentids = df['contentid'].tolist()

    # 모델 초기화 및 코퍼스 추가
    searcher = SemanticSearch()
    searcher.add_corpus(corpus)

    # 검색 수행
    query = "한 남자가 파스타를 먹는다."
    results = searcher.search(query, top_k=5)
    for idx, score in results:
        print(f"contentid: {contentids[idx]}")
        print(f"overview: {corpus[idx]}")
        print(f"score: {score}")
        print("-" * 50)

    # 모델 저장
    searcher.save_model("models/semantic_search.pkl")

    # 나중에 모델 불러오기
    # loaded_searcher = SemanticSearch.load_model("models/semantic_search.pkl")    

if __name__ == "__main__":
    main()