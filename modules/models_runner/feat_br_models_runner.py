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
            LSI_Model_Hyperp.SVD_MODEL_N_COMPONENTS.value: 100,
            LSI_Model_Hyperp.VECTORIZER_NGRAM_RANGE.value: (1,1),
            LSI_Model_Hyperp.VECTORIZER.value : TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True),
            LSI_Model_Hyperp.VECTORIZER_TOKENIZER.value : tok.WordNetBased_LemmaTokenizer()
        }
    
    @staticmethod
    def get_lda_model_hyperp():
        return {
            LDA_Model_Hyperp.LDA_MODEL_N_COMPONENTS.value: 50,
            LDA_Model_Hyperp.LDA_MODEL_RANDOM_STATE.value : 2,
            LDA_Model_Hyperp.VECTORIZER_NGRAM_RANGE.value: (1,1),
            LDA_Model_Hyperp.VECTORIZER.value : TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True),
            LDA_Model_Hyperp.VECTORIZER_TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }
    
    @staticmethod
    def get_bm25_model_hyperp():
        return {
            BM25_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }
    
    @staticmethod
    def get_w2v_hyperp():
        return {
            WordVec_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }

class Feat_BR_Models_Runner:
    def __init__(self):
        self.features_df = fd.Datasets.read_features_df()
        self.bug_reports_df = fd.Datasets.read_selected_bugreports_df()

        self.corpus = self.features_df.feat_desc
        self.query = self.bug_reports_df.br_desc

        self.features_names = self.features_df.feat_name
        self.bug_reports_names = self.bug_reports_df.br_name
        
    def run_lsi_model(self, lsi_hyperp=None):
        print("Running LSI model -----")
        
        if lsi_hyperp == None:
            lsi_hyperp = Feat_BR_Models_Hyperp.get_lsi_model_hyperp()

        lsi_model = LSI(**lsi_hyperp)
        lsi_model.set_name('LSI_Model_Feat_BR')
        lsi_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)
       
        return lsi_model
    
    def run_lda_model(self, lda_hyperp=None):
        print("Running LDA model -----")
        
        if lda_hyperp == None:
            lda_hyperp = Feat_BR_Models_Hyperp.get_lda_model_hyperp()

        lda_model = LDA(**lda_hyperp)
        lda_model.set_name('LDA_Model_Feat_BR')
        lda_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)
        
        return lda_model
    
    def run_bm25_model(self, bm25_hyperp=None):
        print("Running BM25 model -----")
        
        if bm25_hyperp == None:
            bm25_hyperp = Feat_BR_Models_Hyperp.get_bm25_model_hyperp()

        bm25_model = BM_25(**bm25_hyperp)
        bm25_model.set_name('BM25_Model_Feat_BR')
        bm25_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)
        
        return bm25_model
    
    def run_word2vec_model(self, wv_hyperp=None):
        print("Running W2V model -----")
        
        if wv_hyperp == None:
            wv_hyperp = Feat_BR_Models_Hyperp.get_w2v_hyperp()

        wv_model = WordVec_BasedModel(**wv_hyperp)
        wv_model.set_name('WordVec_Model_Feat_BR')
        wv_model.recover_links(self.corpus, self.query, self.features_names, self.bug_reports_names)

        return wv_model