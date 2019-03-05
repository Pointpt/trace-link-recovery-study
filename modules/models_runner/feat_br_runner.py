import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from modules.utils import plots
from modules.utils import firefox_dataset_p2 as fd
from modules.utils import tokenizers as tok
from modules.utils import aux_functions
from modules.utils import model_evaluator as m_eval

from modules.models.lda import LDA
from modules.models.lsi import LSI
from modules.models.bm25 import BM_25
from modules.models.wordvec import WordVec_BasedModel

from modules.models.model_hyperps import LDA_Model_Hyperp
from modules.models.model_hyperps import LSI_Model_Hyperp
from modules.models.model_hyperps import BM25_Model_Hyperp
from modules.models.model_hyperps import WordVec_Model_Hyperp

class Feat_BR_Models_Hyperp:
    
    @staticmethod
    def get_lsi_model_hyperp():
        return {
            LSI_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value : ('cosine' , .80),
            LSI_Model_Hyperp.TOP.value : 100,
            LSI_Model_Hyperp.SVD_MODEL_N_COMPONENTS.value: 100,
            LSI_Model_Hyperp.VECTORIZER_NGRAM_RANGE.value: (1,1),
            LSI_Model_Hyperp.VECTORIZER.value : TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True),
            LSI_Model_Hyperp.VECTORIZER_TOKENIZER.value : tok.WordNetBased_LemmaTokenizer()
        }
    
    @staticmethod
    def get_lda_model_hyperp():
        return {
            LDA_Model_Hyperp.TOP.value : 100,
            LDA_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value : ('cosine',.75),
            LDA_Model_Hyperp.LDA_MODEL_N_COMPONENTS.value: 50,
            LDA_Model_Hyperp.LDA_MODEL_RANDOM_STATE.value : 2,
            LDA_Model_Hyperp.VECTORIZER_NGRAM_RANGE.value: (1,1),
            LDA_Model_Hyperp.VECTORIZER.value : TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True),
            LDA_Model_Hyperp.VECTORIZER_TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }
    
    @staticmethod
    def get_bm25_model_hyperp():
        return {
            BM25_Model_Hyperp.TOP.value : 100,
            BM25_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value : ('-', 0.0),
            BM25_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }
    
    @staticmethod
    def get_w2v_hyperp():
        return {
            WordVec_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value : ('cosine', .80),
            WordVec_Model_Hyperp.TOP.value : 100,
            WordVec_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }

class Feat_BR_Runner:
    def __init__(self):
        self.features_df = fd.Datasets.read_features_df()
        self.bug_reports_df = fd.Datasets.read_selected_bug_reports_2_df()

        self.corpus = self.features_df.feat_desc
        self.query = self.bug_reports_df.br_desc

        self.features_names = self.features_df.feat_name
        self.bug_reports_names = self.bug_reports_df.br_name

        self.orc = fd.Feat_BR_Oracles.read_feat_br_expert_volunteers_df()


    def run_lsi_model(self, lsi_hyperp=None):
        if lsi_hyperp == None:
            lsi_hyperp = Feat_BR_Models_Hyperp.get_lsi_model_hyperp()

        lsi_model = LSI(**lsi_hyperp)
        lsi_model.set_name('LSI_Model_Feat_BR')
        lsi_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)

        print("\nModel Evaluation -------------------------------------------")
        evaluator = m_eval.ModelEvaluator(self.orc, lsi_model)
        evaluator.evaluate_model(verbose=True)
        
        return (lsi_model, evaluator)
    
    def run_lda_model(self, lda_hyperp=None):
        if lda_hyperp == None:
            lda_hyperp = Feat_BR_Models_Hyperp.get_lda_model_hyperp()

        lda_model = LDA(**lda_hyperp)
        lda_model.set_name('LDA_Model_Feat_BR')
        lda_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)

        print("\nModel Evaluation -------------------------------------------")
        evaluator = m_eval.ModelEvaluator(self.orc, lda_model)
        evaluator.evaluate_model(verbose=True)
        
        return (lda_model, evaluator)
    
    def run_bm25_model(self, bm25_hyperp=None):
        if bm25_hyperp == None:
            bm25_hyperp = Feat_BR_Models_Hyperp.get_bm25_model_hyperp()
        bm25_hyperp = {
            BM25_Model_Hyperp.TOP.value : 100,
            BM25_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value : ('-', 0.0),
            BM25_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }

        bm25_model = BM_25(**bm25_hyperp)
        bm25_model.set_name('BM25_Model_Feat_BR')
        bm25_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)

        print("\nModel Evaluation -------------------------------------------")
        evaluator = m_eval.ModelEvaluator(self.orc, bm25_model)
        evaluator.evaluate_model(verbose=True)
        
        return (bm25_model, evaluator)
    
    def run_word2vec_model(self, wv_hyperp=None):
        if wv_hyperp == None:
            wv_hyperp = Feat_BR_Models_Hyperp.get_w2v_hyperp()

        wv_model = WordVec_BasedModel(**wv_hyperp)
        wv_model.set_name('WordVec_Model_Feat_BR')
        wv_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)

        print("\nModel Evaluation -------------------------------------------")
        evaluator = m_eval.ModelEvaluator(self.orc, wv_model)
        evaluator.evaluate_model(verbose=True)
        
        return (wv_model, evaluator)