
import pandas as pd
import codecs
import os


BASE_DIR = 'data/jEdit/jEditDataset/'
                
class OracleLoader():

    def __init__(self):
        self.files = ['BR_UC']
        self.trace_dict = {}
        self.list_source_artfs = ['BR_{}_SRC'.format(f.replace('.txt','')) for f in os.listdir(BASE_DIR + "/docs_english/BR/") if os.path.isfile(BASE_DIR + "/docs_english/BR/" + f)]
        self.list_target_artfs = ['UC_{}_TRG'.format(f.replace('.txt','')) for f in os.listdir(BASE_DIR + "/docs_english/UC/") if os.path.isfile(BASE_DIR + "/docs_english/UC/" + f)]

    def load_oracle(self):
        for f_name in self.files:
            with open(BASE_DIR + 'oracle/' + f_name + '.txt', 'r') as fl:
                lines = fl.readlines()
                source_abv = f_name.split('_')[0] + '_'
                target_abv = f_name.split('_')[1] + '_'
                for l in lines:
                    source, targets = l.split(':')[0], l.split(':')[1]
                    targets_list = targets.split(' ')
                    
                    s_name = source_abv + source.replace('.txt', '') + '_SRC'
                    t_list_names = [target_abv  + t.replace('.txt', '').replace('\n','') + '_TRG' for t in targets_list]

                    self.trace_dict[s_name] = t_list_names

    def save_oracle_as_csv(self):        
        print("list_source_artfs: " + str(self.list_source_artfs))
        print("list_target_artfs: " + str(self.list_target_artfs))
        
        rows_list = []
        for src_art in self.list_source_artfs:
            for trg_art in self.list_target_artfs:
                if trg_art in self.trace_dict[src_art]:
                    rows_list.append({'src_artf': src_art, 'trg_artf': trg_art, 'link': 1})
                else:
                    rows_list.append({'src_artf': src_art, 'trg_artf': trg_art, 'link': 0})
        
        self.df_trace = pd.DataFrame(rows_list, columns=['src_artf', 'trg_artf', 'link'])
        self.df_trace.to_csv(BASE_DIR + '/oracle/output/trace_matrix.csv', sep=',', index=False)


    def save_artfs_descriptions(self):
        df_artifacts_descriptions = pd.DataFrame(columns=['artf_name','artf_description'])

        for list_artfs in [self.list_source_artfs, self.list_target_artfs]:
            for artf in list_artfs:
                artf_type = artf.split('_')[0]   # UC, CC, ID, TC
                artf_num = artf.split('_')[1]    # 1, 2, 3, ...

                new_row = []
                with codecs.open('{0}{1}/{2}/{3}.txt'.format(BASE_DIR, 'docs_english', artf_type, artf_num), encoding='utf-8') as fl:
                    new_row.append({'artf_name' : artf, 'artf_description' : fl.read()})
                    df_artifacts_descriptions = df_artifacts_descriptions.append(new_row, sort=False)    

        df_artifacts_descriptions.to_csv(BASE_DIR + '/oracle/output/artifacts_descriptions.csv', sep='|', index=False)

if __name__ == '__main__':    
    oLoader = OracleLoader()
    oLoader.load_oracle()
    oLoader.save_oracle_as_csv()
    oLoader.save_artfs_descriptions()
