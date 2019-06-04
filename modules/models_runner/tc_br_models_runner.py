import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from modules.utils import plots
from modules.utils import firefox_dataset_p2 as fd
from modules.utils import tokenizers as tok
from modules.utils import aux_functions

from modules.models.lda import LDA
from modules.models.lsi import LSI
from modules.models.bm25 import BM_25
from modules.models.wordvec import WordVec_BasedModel
from modules.models.zeror import ZeroR_Model

import modules.models.model_hyperps as mh

class TC_BR_Models_Hyperp:
    
    @staticmethod
    def get_lsi_model_hyperp():
        return {
            mh.LSI_Model_Hyperp.SVD_MODEL_N_COMPONENTS.value: 20,
            mh.LSI_Model_Hyperp.VECTORIZER_NGRAM_RANGE.value: (1,1),
            mh.LSI_Model_Hyperp.VECTORIZER_MAX_FEATURES.value: 400,
            mh.LSI_Model_Hyperp.VECTORIZER.value : TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True),
            mh.LSI_Model_Hyperp.VECTORIZER_TOKENIZER.value : tok.WordNetBased_LemmaTokenizer()
        }
    
    @staticmethod
    def get_lda_model_hyperp():
        return {
            mh.LDA_Model_Hyperp.LDA_MODEL_N_COMPONENTS.value: 20,
            mh.LDA_Model_Hyperp.LDA_MODEL_RANDOM_STATE.value : 2,
            mh.LDA_Model_Hyperp.VECTORIZER_NGRAM_RANGE.value: (1,1),
            mh.LDA_Model_Hyperp.VECTORIZER_MAX_FEATURES.value: 200,
            mh.LDA_Model_Hyperp.VECTORIZER.value : TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True),
            mh.LDA_Model_Hyperp.VECTORIZER_TOKENIZER.value : tok.PorterStemmerBased_Tokenizer() 
        }
    
    @staticmethod
    def get_bm25_model_hyperp():
        return {
            mh.BM25_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }
    
    @staticmethod
    def get_w2v_model_hyperp():
        return {
            mh.WordVec_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }

class TC_BR_Runner:
    def __init__(self, testcases=pd.DataFrame(), bugreports=pd.DataFrame()):
        self.test_cases_df = None
        self.bug_reports_df = None
        self.corpus = None
        self.query = None
        self.test_cases_names = None
        self.bug_reports_names = None
        
        self.set_basic_params(testcases, bugreports)
        
    
    def set_basic_params(self, testcases, bugreports):
        if testcases.empty:
            self.test_cases_df = fd.Datasets.read_testcases_df()
        else:
            self.test_cases_df = testcases
        
        if bugreports.empty:
            self.bug_reports_df = fd.Datasets.read_selected_bugreports_df()
        else:
            self.bug_reports_df = bugreports
        
        self.corpus = self.test_cases_df.tc_desc
        self.query = self.bug_reports_df.br_desc
        
        self.test_cases_names = self.test_cases_df.TC_Number
        self.bug_reports_names = self.bug_reports_df.Bug_Number
    
    def run_lsi_model(self, lsi_hyperp=None):
        print("Running LSI Model ------")
        
        if lsi_hyperp == None:
            lsi_hyperp = TC_BR_Models_Hyperp.get_lsi_model_hyperp()

        lsi_model = LSI(**lsi_hyperp)
        lsi_model.set_name('LSI_Model_TC_BR')
        
        lsi_model.recover_links(self.corpus, self.query, self.test_cases_names, self.bug_reports_names)
        
        return lsi_model
    
    def run_lda_model(self, lda_hyperp=None):
        print("Running LDA Model -----")
        
        if lda_hyperp == None:
            lda_hyperp = TC_BR_Models_Hyperp.get_lda_model_hyperp()

        lda_model = LDA(**lda_hyperp)
        lda_model.set_name('LDA_Model_TC_BR')
        lda_model.recover_links(self.corpus, self.query, self.test_cases_names, self.bug_reports_names)

        return lda_model
    
    def run_bm25_model(self, bm25_hyperp=None):
        print("Running BM25 Model -----")
        
        if bm25_hyperp == None:
            bm25_hyperp = TC_BR_Models_Hyperp.get_bm25_model_hyperp()

        bm25_model = BM_25(**bm25_hyperp)
        bm25_model.set_name('BM25_Model_TC_BR')
        bm25_model.recover_links(self.corpus, self.query, self.test_cases_names, self.bug_reports_names)

        return bm25_model
    
    def run_word2vec_model(self, wv_hyperp=None):
        print("Running W2V Model ------")
        
        if wv_hyperp == None:
            wv_hyperp = TC_BR_Models_Hyperp.get_w2v_model_hyperp()

        wv_model = WordVec_BasedModel(**wv_hyperp)
        wv_model.set_name('WordVec_Model_TC_BR')
        wv_model.recover_links(self.corpus, self.query, self.test_cases_names, self.bug_reports_names)

        return wv_model
    
    def run_zeror_model(self, zeror_hyperp=None):
        print("Running ZeroR model -----")
        
        oracle = fd.Tc_BR_Oracles.read_oracle_expert_volunteers_intersec_df()
        
        zeror_model = ZeroR_Model(oracle)
        zeror_model.set_name('ZeroR_Model_TC_BR')
        zeror_model.recover_links()
        
        return zeror_model