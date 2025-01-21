# tests/bert/kobert_batch1.py
import os
import pandas as pd
import torch
import numpy as np
from kobert_transformers import get_kobert_model, get_tokenizer
from threading import Thread
import gc
import logging
import time
import sys
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from konlpy.tag import Okt

# 로딩 애니메이션 핸들러
class LoadingAnimationHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.animation_chars = "|/-\\"
        self.current_idx = 0
        self.is_running = False
        self.thread = None
        self.last_message = ""
    
    def emit(self, record):
        if not self.is_running: 
            return
        self.last_message = record.getMessage()
    
    def start(self):
        self.is_running = True
        self.thread = Thread(target=self.animate, daemon=True)
        self.thread.start()
    
    def animate(self):
        while self.is_running:
            sys.stdout.write(f"\r{self.last_message} {self.animation_chars[self.current_idx]}")
            sys.stdout.flush()
            self.current_idx = (self.current_idx +1) % len(self.animation_chars)
            time.sleep(0.2)
    
    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join()
        sys.stdout.write("\nComplete!\n")
        sys.stdout.flush()

# 배치 로더 클래스
class BatchLoader():
    def __init__(self, csv_path: str, keyword_path: str, working_batch_size: int=16):
        self.csv_path = csv_path
        self.keyword_path = keyword_path
        self.working_batch_size = working_batch_size

        if not os.path.exists(csv_path):
            raise FileNotFoundError (f"\n[ERROR] BatchLoader failed to find any .csv files.\n")
        
        # .csv 파일 이름 목록
        self.df_list = sorted([
            entry.name for entry in os.scandir(csv_path)
            if entry.is_file() and entry.name.endswith('.csv')
        ])
        self.current_idx = 0

        if not os.path.exists(keyword_path):
            os.makedirs(keyword_path, exist_ok=True)
            logging.info(f"\n[BatchLoader] keyword destination path {keyword_path} created.\n")
            logging.info(f"\n[BatchLoader] new path: starting from batch 0.\n")
        else:
            logging.info(f"\n[BatchLoader] successfully found keyword destination path {keyword_path}.\n")
            discovered = len([
                f for f in os.listdir(keyword_path)
                if f.endswith('.parquet')
            ])
            if discovered > 0:
                logging.info(f"\n[BatchLoader] discovered {discovered} number of completed .parquet files.\n")
                logging.info(f"\n[BatchLoader] updating current working batch to {discovered}.\n")
                self.current_idx = discovered
            else:
                logging.info(f"\n[BatchLoader] did not find any complete batches. starting from 0.\n")

    # 현��� 배치의 DataFrame을 반환
    def fetch_current_df(self):
        if self.current_idx >= len(self.df_list):
            return None  # CSV 파일의 끝에 도달
        try:
            df = pd.read_csv(os.path.join(self.csv_path, self.df_list[self.current_idx]))
            return df[[
                'contentid',
                'title',
                'overview',
                'cat3',
                'region'
            ]]
        except FileNotFoundError:
            logging.error(f"\n[ERROR] could not find file {self.df_list[self.current_idx]}. It really should exist.")
            return pd.DataFrame()  # 빈 DataFrame 반환
    
    # 배치 인덱스 업데이트
    def update(self):
        self.current_idx += 1

# KoBERT 임베딩 추출 클래스
class KobertExtractor:
    def __init__(self, keyword_path: str, top_n: int=10, batch_size: int=8):
        self.keyword_path = keyword_path
        self.top_n = top_n
        self.batch_size = batch_size
        self.model = get_kobert_model()
        self.tokenizer = get_tokenizer()
        
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # KoNLPy Okt 객체 초기화
        self.okt = Okt()

        # 한국어 불용어 리스트 정의
        self.stop_words = set([
            '그', '저', '이', '것', '그것', '저것', '여기', '거기', '저기',
            '나', '너', '그녀', '그들', '우리', '너희', '저희', '의', '의해서',
            '위해', '에게', '으로', '에서', '한테', '하고', '하고는', '하고도',
            '그리고', '하지만', '그러나', '그래서', '그러므로'
            # 필요한 만큼 추가
        ])
        self.lemmatizer = WordNetLemmatizer()

    # 텍스트 전처리 메서드
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z가-힣\s]', '', text)
        tokens = self.okt.morphs(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        logging.info(f"[KobertExtractor] Preprocessed text: {text}.")
        return ' '.join(tokens)

    # 배치 처리 메서드
    def process_batch(self, batch: pd.DataFrame):
        texts = [
            text for text in batch['overview'].tolist() 
            if isinstance(text, str) and len(text.strip()) > 0
        ]
        if not texts:
            logging.warning("Skipping batch with no valid text data.")
            return []
        
        # 텍스트 전처리
        processed_texts = [self.preprocess_text(text) for text in texts]
        logging.info(f"[KobertExtractor] Processed {len(processed_texts)} texts for embedding.")
        
        inputs = self.tokenizer(processed_texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 임베딩 추출 ([CLS] 토큰의 출력)
        cls_embeddings = outputs[0][:,0,:].cpu().numpy()
        logging.info(f"[KobertExtractor] Extracted embeddings for batch.")
        torch.cuda.empty_cache()
        return cls_embeddings

    # 임베딩 저장 메서드
    def save_embeddings(self, embeddings: np.ndarray, batch_idx: int):
        df = pd.DataFrame(embeddings)
        df.to_parquet(os.path.join(self.keyword_path, f"batch_{batch_idx}.parquet"), index=False)
        logging.info(f"[KobertExtractor] Saved embeddings for batch {batch_idx}.")
        
        # 메모리 해제를 위해 변수 삭제 및 가비지 컬렉션 실행
        del df, embeddings
        gc.collect()

    # 키워드 추출 메서드 (추후 활용 가능)
    def extract_keywords(self, df: pd.DataFrame):
        # 필요 시 구현
        pass

def main():
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    handler = LoadingAnimationHandler()
    logger.addHandler(handler)

    CSV_PATH ='./batch/'    
    KEYWORD_PATH = './keywords'

    try:
        batch_loader = BatchLoader(csv_path=CSV_PATH, keyword_path=KEYWORD_PATH)       
        handler.start()
        logger.info("Processing...")
        
        kobert_extractor = KobertExtractor(keyword_path=KEYWORD_PATH, top_n=10, batch_size=16)

        while True:
            df = batch_loader.fetch_current_df()
            if df is None:
                break
            embeddings = kobert_extractor.process_batch(df)
            if len(embeddings) > 0:
                kobert_extractor.save_embeddings(embeddings, batch_loader.current_idx)
            batch_loader.update()
            
            # 메모리 해제를 위해 변수 삭제 및 가비지 컬렉션 실행
            del df, embeddings
            gc.collect()
        
        logging.info(f"\n[DEBUG] Processed {batch_loader.current_idx} number of .csv files.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        handler.stop()

def test_single_batch():
    # 테스트용 단일 배치 처리
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    handler = LoadingAnimationHandler()
    logger.addHandler(handler)

    CSV_PATH ='./'
    KEYWORD_PATH = './keywords'

    try:
        batch_loader = BatchLoader(csv_path=CSV_PATH, keyword_path=KEYWORD_PATH)
        handler.start()
        logger.info("Processing a single batch...")

        df = batch_loader.fetch_current_df()
        if df is not None:
            kobert_extractor = KobertExtractor(keyword_path=KEYWORD_PATH, top_n=5, batch_size=8)
            embeddings = kobert_extractor.process_batch(df.head(8))  # 첫 8개 행 처리
            kobert_extractor.save_embeddings(embeddings, batch_loader.current_idx)
            logger.info(f"[TEST] Extracted and saved embeddings for batch {batch_loader.current_idx}.")
        
            # 메모리 해제를 위해 변수 삭제 및 가비지 컬렉션 실행
            del df, embeddings
            gc.collect()
        else:
            logger.error("No data found in the first batch.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        handler.stop()

if __name__ == "__main__":
    # main()
    test_single_batch()