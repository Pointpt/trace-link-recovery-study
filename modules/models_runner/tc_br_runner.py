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


class TC_BR_Runner:
    def __init__(self):
        self.test_cases_df = fd.Datasets.read_testcases_df()
        self.bug_reports_df = fd.Datasets.read_selected_bugreports_df()
        
        self.corpus = self.test_cases_df.tc_desc
        self.query = self.bug_reports_df.br_desc

        self.test_cases_names = self.test_cases_df.tc_name
        self.bug_reports_names = self.bug_reports_df.br_name

        self.orc = fd.Tc_BR_Oracles.read_oracle_expert_volunteers_intersec_df()


    def run_lsi_model(self):
        lsi_hyperp = {
            LSI_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value : ('cosine' , .80),
            LSI_Model_Hyperp.TOP.value : 15,
            LSI_Model_Hyperp.SVD_MODEL_N_COMPONENTS.value: 100,
            LSI_Model_Hyperp.VECTORIZER_NGRAM_RANGE.value: (1,1),
            LSI_Model_Hyperp.VECTORIZER.value : TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True),
            LSI_Model_Hyperp.VECTORIZER_TOKENIZER.value : tok.WordNetBased_LemmaTokenizer()
        }

        lsi_model = LSI(**lsi_hyperp)
        lsi_model.set_name('LSI_Model_TC_BR')
        lsi_model.recover_links(self.corpus, self.query, self.test_cases_names, self.bug_reports_names)

        print("\nModel Evaluation -------------------------------------------")
        evaluator = m_eval.ModelEvaluator(self.orc, lsi_model)
        evaluator.evaluate_model(verbose=True)
        
        return (lsi_model, evaluator)
    
    def run_lda_model(self):
        lda_hyperp = {
            LDA_Model_Hyperp.TOP.value : 15,
            LDA_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value : ('cosine',.75),
            LDA_Model_Hyperp.LDA_MODEL_N_COMPONENTS.value: 20,
            LDA_Model_Hyperp.LDA_MODEL_RANDOM_STATE.value : 2,
            LDA_Model_Hyperp.VECTORIZER_NGRAM_RANGE.value: (1,1),
            LDA_Model_Hyperp.VECTORIZER.value : TfidfVectorizer(stop_words='english', use_idf=True, smooth_idf=True),
            LDA_Model_Hyperp.VECTORIZER_TOKENIZER.value : tok.PorterStemmerBased_Tokenizer() 
        }

        lda_model = LDA(**lda_hyperp)
        lda_model.set_name('LDA_Model_TC_BR')
        lda_model.recover_links(self.corpus, self.query, self.test_cases_names, self.bug_reports_names)

        print("\nModel Evaluation -------------------------------------------")
        evaluator = m_eval.ModelEvaluator(self.orc, lda_model)
        evaluator.evaluate_model(verbose=True)
        
        return (lda_model, evaluator)
    
    def run_bm25_model(self):
        bm25_hyperp = {
            BM25_Model_Hyperp.TOP.value : 15,
            BM25_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value : ('-', 0.0),
            BM25_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }

        bm25_model = BM_25(**bm25_hyperp)
        bm25_model.set_name('BM25_Model_TC_BR')
        bm25_model.recover_links(self.corpus, self.query, self.test_cases_names, self.bug_reports_names)

        print("\nModel Evaluation -------------------------------------------")
        evaluator = m_eval.ModelEvaluator(self.orc, bm25_model)
        evaluator.evaluate_model(verbose=True)
        
        return (bm25_model, evaluator)
    
    def run_word2vec_model(self):
        wv_hyperp = {
            WordVec_Model_Hyperp.SIM_MEASURE_MIN_THRESHOLD.value : ('cosine', .80),
            WordVec_Model_Hyperp.TOP.value : 15,
            WordVec_Model_Hyperp.TOKENIZER.value : tok.PorterStemmerBased_Tokenizer()
        }

        wv_model = WordVec_BasedModel(**wv_hyperp)
        wv_model.set_name('WordVec_Model_TC_BR')
        wv_model.recover_links(self.corpus, self.query, self.test_cases_names, self.bug_reports_names)

        print("\nModel Evaluation -------------------------------------------")
        evaluator = m_eval.ModelEvaluator(self.orc, wv_model)
        evaluator.evaluate_model(verbose=True)
        
        return (wv_model, evaluator)