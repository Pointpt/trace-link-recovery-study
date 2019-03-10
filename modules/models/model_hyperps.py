from enum import Enum

class LSI_Model_Hyperp(Enum):
    NAME = 'lsi__name'
    SIMILARITY_MEASURE = 'lsi__similarity_measure'
    VECTORIZER = 'lsi__vectorizer'
    VECTORIZER_STOP_WORDS = 'lsi__vectorizer__stop_words'
    VECTORIZER_TOKENIZER = 'lsi__vectorizer__tokenizer'
    VECTORIZER_USE_IDF = 'lsi__vectorizer__use_idf'
    VECTORIZER_SMOOTH_IDF = 'lsi__vectorizer__smooth_idf'
    VECTORIZER_NGRAM_RANGE = 'lsi__vectorizer__ngram_range'
    SVD_MODEL = 'lsi__svd_model'
    SVD_MODEL_N_COMPONENTS = 'lsi__svd_model__n_components'

    
class LDA_Model_Hyperp(Enum):
    NAME = 'lda__name'
    SIMILARITY_MEASURE = 'lda__similarity_measure'
    VECTORIZER = 'lda__vectorizer'
    VECTORIZER_STOP_WORDS = 'lda__vectorizer__stop_words'
    VECTORIZER_TOKENIZER = 'lda__vectorizer__tokenizer'
    VECTORIZER_USE_IDF = 'lda__vectorizer__use_idf'
    VECTORIZER_SMOOTH_IDF = 'lda__vectorizer__smooth_idf'
    VECTORIZER_NGRAM_RANGE = 'lda__vectorizer__ngram_range'
    LDA_MODEL = 'lda__lda_model'
    LDA_MODEL_N_COMPONENTS = 'lda__lda_model__n_components'
    LDA_MODEL_RANDOM_STATE = 'lda__lda_model__random_state'
    TOKENIZER = 'lda__tokenizer'

    
class BM25_Model_Hyperp(Enum):
    NAME = 'bm25__name'
    K = 'bm25__k'
    B = 'bm25__b'
    EPSILON = 'bm25__epsilon'
    TOKENIZER = 'bm25__tokenizer'
    
    
class WordVec_Model_Hyperp(Enum):
    NAME = 'wordvec__name'
    TOKENIZER = 'wordvec__tokenizer'
    