# KoBERT를 이용한 임베딩 로컬 실행 코드 작성

아래는 `tests/bert/kobert_batch1.py` 파일을 활용하여 KoBERT 모델을 사용해 `overview` 컬럼의 텍스트 데이터를 임베딩하는 로컬 실행 코드를 작성한 예제입니다. 이 코드는 배치 단위로 데이터를 처리하고, 각 텍스트의 임베딩을 추출하여 로컬에 저장하는 기능을 포함하고 있습니다.

## 1. 환경 설정

### 필요한 라이브러리 설치

먼저, 필요한 라이브러리를 설치합니다. KoBERT와 관련된 패키지 외에도 데이터 처리를 위한 `pandas`, 텍스트 전처리를 위한 `nltk`, 임베딩 생성을 위한 `torch` 등이 필요합니다.

```bash
pip install pandas torch transformers kobert-transformers nltk scikit-learn
```

### NLTK 리소스 다운로드

텍스트 전처리를 위해 NLTK의 불용어와 토크나이저를 사용하므로 필요한 리소스를 다운로드합니다.

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 2. `kobert_batch1.py` 파일 구성

아래는 `kobert_batch1.py` 파일의 완전한 예제 코드입니다. 이 코드는 CSV 파일을 로드하고, 배치 단위로 텍스트 데이터를 전처리한 후 KoBERT 모델을 이용해 임베딩을 생성합니다.

```python
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

    # 현재 배치의 DataFrame을 반환
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

        # 전처리 도구 초기화
        self.stop_words = set(stopwords.words('korean'))
        self.lemmatizer = WordNetLemmatizer()

    # 텍스트 전처리 메서드
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z가-힣\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
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
        
        inputs = self.tokenizer(processed_texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 임베딩 추출 ([CLS] 토큰의 출력)
        cls_embeddings = outputs[0][:,0,:].cpu().numpy()
        return cls_embeddings

    # 임베딩 저장 메서드
    def save_embeddings(self, embeddings: np.ndarray, batch_idx: int):
        df = pd.DataFrame(embeddings)
        df.to_parquet(os.path.join(self.keyword_path, f"batch_{batch_idx}.parquet"), index=False)
        logging.info(f"[KobertExtractor] Saved embeddings for batch {batch_idx}.")

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

    CSV_PATH ='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'
    KEYWORD_PATH = '/home/nishtala/TripCok/TripCok_models/src/tripcok_models/models/parquet/keywords'

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

    CSV_PATH ='/home/nishtala/TripCok/TripCok_models/src/tripcok_models/csv_maker/batch/'
    KEYWORD_PATH = '/home/nishtala/TripCok/TripCok_models/src/tripcok_models/models/parquet/keywords'

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
        else:
            logger.error("No data found in the first batch.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        handler.stop()

if __name__ == "__main__":
    # main()
    test_single_batch()
```

## 3. 코드 설명

### 3.1. 로딩 애니메이션 핸들러

- **`LoadingAnimationHandler` 클래스**: 로딩 중임을 사용자에게 시각적으로 알려주는 애니메이션을 출력합니다. `logging` 모듈과 연동하여 로그 메시지와 함께 애니메이션을 표시합니다.

### 3.2. 배치 로더

- **`BatchLoader` 클래스**: 지정된 디렉토리에서 CSV 파일을 로드하고, 현재 처리 중인 배치를 관리합니다.
  - **`fetch_current_df` 메서드**: 현재 인덱스에 해당하는 CSV 파일을 읽어와 필요한 컬럼만 반환합니다.
  - **`update` 메서드**: 다음 배치로 인덱스를 증가시킵니다.

### 3.3. KoBERT 임베딩 추출기

- **`KobertExtractor` 클래스**: KoBERT 모델을 사용하여 텍스트의 임베딩을 추출합니다.
  - **`preprocess_text` 메서드**: 텍스트를 소문자로 변환하고, 특수문자를 제거한 후 토큰화하고 불용어를 제거합니다.
  - **`process_batch` 메서드**: 배치 단위로 전처리된 텍스트를 KoBERT에 입력하여 임베딩을 추출합니다.
  - **`save_embeddings` 메서드**: 추출된 임베딩을 Parquet 파일로 저장합니다.

### 3.4. 메인 함수

- **`main` 함수**: 전체 CSV 파일을 배치 단위로 처리하여 임베딩을 추출하고 저장합니다.
  - 로딩 애니메이션을 시작하고, 모든 CSV 파일을 순차적으로 처리합니다.
  - 각 배치의 임베딩을 `keywords` 디렉토리에 Parquet 파일로 저장합니다.

### 3.5. 테스트 함수

- **`test_single_batch` 함수**: 단일 배치를 테스트하기 위한 함수입니다.
  - 첫 번째 배치를 처리하고 임베딩을 저장합니다.
  - 이를 통해 전체 프로세스가 정상적으로 동작하는지 확인할 수 있습니다.

## 4. 실행 방법

1. **CSV 파일 준비**: `CSV_PATH`에 지정된 디렉토리에 `overview` 컬럼을 포함한 CSV 파일들을 준비합니다.

2. **코드 실행**: 터미널에서 다음 명령어를 실행하여 단일 배치 테스트를 수행합니다.

    ```bash
    python tests/bert/kobert_batch1.py
    ```

    만약 모든 배치를 처리하고 싶다면 `main()` 함수를 호출하도록 `__main__` 블록을 수정합니다.

    ```python
    if __name__ == "__main__":
        main()
        # test_single_batch()
    ```

    그리고 다시 실행합니다.

    ```bash
    python tests/bert/kobert_batch1.py
    ```

3. **임베딩 결과 확인**: 지정된 `KEYWORD_PATH` 디렉토리에 각 배치별로 `batch_{n}.parquet` 파일이 생성됩니다. 이 파일들은 각 배치의 임베딩 벡터를 포함하고 있습니다.

## 5. 추가 고려 사항

- **GPU 사용**: KoBERT 모델은 GPU를 활용할 수 있습니다. GPU를 사용하려면 CUDA가 설치된 환경에서 실행해야 하며, 코드에서는 자동으로 `cuda`를 감지하여 사용하도록 설정되어 있습니다.

- **메모리 관리**: 대용량 데이터를 처리할 경우 메모리 관리가 중요합니다. 배치 단위로 데이터를 처리하고, 각 배치가 끝난 후에는 메모리를 해제하도록 `del`과 `gc.collect()`를 활용할 수 있습니다.

- **에러 핸들링**: 코드에는 다양한 예외 처리가 포함되어 있어, 파일이 누락되거나 데이터가 비어있는 경우에도 안정적으로 작동하도록 설계되었습니다.

- **키워드 추출**: 현재 코드는 임베딩을 추출하고 저장하는 기능에 집중되어 있습니다. 추후 임베딩을 기반으로 유사한 콘텐츠를 추천하는 로직을 추가할 수 있습니다.

## 6. 결론

위의 코드를 통해 KoBERT를 이용하여 `overview` 텍스트 데이터를 임베딩하고, 이를 로컬에서 효과적으로 처리할 수 있습니다. 배치 단위로 데이터를 처리함으로써 대용량 데이터도 효율적으로 관리할 수 있으며, 추출된 임베딩을 활용하여 콘텐츠 기반 추천 시스템을 구축하는 데 기초 자료로 활용할 수 있습니다.
