import pandas as pd
from abc import ABCMeta, abstractmethod

class GenericModel(metaclass=ABCMeta):
    def __init__(self):
        self.name = None
        self.top = None
        self.sim_measure_min_threshold = None
        
        self.trace_links_df = None
        self.model_dump_path = None
        self.model_gen_name = None    
    
    def set_name(self, name):
        self.name = name
    
    def set_model_gen_name(self, gen_name):
        self.model_gen_name = gen_name
    
    def set_top(self, top):
        self.top = top
    
    def set_sim_measure_min_threshold(self, sim_measure_min_threshold):
        self.sim_measure_min_threshold = sim_measure_min_threshold
    
    @abstractmethod
    def recover_links(self, corpus, query, use_cases_names, bug_reports_names):
        pass
    
    def _fillUp_traceLinksDf(self, test_cases_names, bug_reports_names, sim_matrix):
        self.trace_links_df = pd.DataFrame(index = test_cases_names,
                                           columns = bug_reports_names,
                                           data = sim_matrix)
                    
        for col in self.trace_links_df.columns:
            nlargest_df = self.trace_links_df.nlargest(n = self.top, columns=col, keep='first')    
            self.trace_links_df[col] = [1 if x in nlargest_df[col].tolist() and x >= self.sim_measure_min_threshold[1] else 0 for x in self.trace_links_df[col]]

    def save_sim_matrix(self):
        self._sim_matrix.to_csv('models_sim_matrix/{}.csv'.format(self.get_model_gen_name()))
    
    def save_trace_matrix(self):
        self.trace_links_df.to_csv('models_trace_matrix/{}.csv'.format(self.get_model_gen_name()))
    
    def get_name(self):
        return self.name
    
    def get_top_value(self):
        return self.top
    
    def get_sim_measure_min_threshold(self):
        return self.sim_measure_min_threshold
    
    def get_sim_matrix(self):
        return self._sim_matrix
    
    def get_trace_links_df(self):
        return self.trace_links_df
    
    def get_model_dump_path(self):
        return 'dumps/{}/model/{}.p'.format(self.get_model_gen_name(), self.get_name())
                                
    def get_model_gen_name(self):
        return self.model_gen_name
    
    @abstractmethod
    def model_setup(self):
        pass