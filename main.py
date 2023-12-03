import gensim
import logging
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
# 로깅 설정
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def preprocess_english_text(text):
    # 단어 토큰화
    words = word_tokenize(text)
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.lower() not in stop_words]
    return words

# 논문파일 문장 읽기 및 전처리
def read_and_preprocess_sentences(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            processed_line = preprocess_english_text(line.strip())
            if processed_line:  # 빈 문장 제외
                sentences.append(processed_line)
    return sentences

# 파일 경로 설정
file_path = 'corpus.txt'

# 파일에서 문장 읽기 및 전처리
sentences = read_and_preprocess_sentences(file_path)

# Word2Vec 모델 학습
model = gensim.models.Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# 벡터와 메타데이터 파일 내보내기 (TensorFlow Projector용)
with open("word_vectors.tsv", 'w', encoding='utf-8') as file_vectors, open("metadata.tsv", 'w', encoding='utf-8') as file_metadata:
    for word in model.wv.index_to_key:
        file_metadata.write(word + '\n')
        file_vectors.write('\t'.join([str(x) for x in model.wv[word]]) + '\n')
