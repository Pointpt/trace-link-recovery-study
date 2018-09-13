
import pandas as pd

files = ['CC_CC', 'CC_TC', 
            'ID_TC', 'ID_ID', 'ID_CC', 'ID_UC',
            'TC_CC', 'TC_TC', 
            'UC_CC', 'UC_ID', 'UC_TC', 'UC_UC']

trace_dict = {}
df_trace = None

sources_names = []
targets_names = []

df_artifacts_descriptions = pd.DataFrame(columns=['artf_name','description'])

BASE_DIR = 'data/EasyClinic/EasyClinicDataset/'


def _register_source_artifact(s_name):
    if s_name not in sources_names:
        sources_names.append(s_name)
    
    global df_artifacts_descriptions

    if s_name not in list(df_artifacts_descriptions.artf_name):
        new_row = []
        
        artf_type = s_name.split('_')[0]
        artf_num = s_name.split('_')[1]

        fl = open('{0}{1}/{2}/{3}.txt'.format(BASE_DIR, 'docs_english', artf_type, artf_num))
        artf_desc = fl.read().encode('utf-8')
        d = {s_name : artf_desc}

        new_row.append(d)
    
    df_artifacts_descriptions = df_artifacts_descriptions.append(new_row, sort=True)


def _register_target_artifact(t_list_names):
    for t_name in t_list_names:
        if t_name not in targets_names:
            targets_names.append(t_name)


def read_oracle_files():
    for f_name in files:
        fl = open(BASE_DIR + 'oracle/' + f_name + '.txt')
        lines = fl.readlines()
        source_abv = f_name.split('_')[0] + '_'
        target_abv = f_name.split('_')[1] + '_'
        for l in lines:
            source, targets = l.split(':')[0], l.split(':')[1]
            targets_list = targets.split(' ')
            
            s_name = source_abv + source.replace('.txt', '') + '_SRC'
            t_list_names = [target_abv  + t.replace('.txt', '') + '_TRG' for t in targets_list[:-1]]

            trace_dict[s_name] = t_list_names

            _register_source_artifact(s_name)
            _register_target_artifact(t_list_names)


def dict2df():
    rows_list = []
    for artf_1 in sources_names:
        for artf_2 in targets_names:
            if artf_2.split('_')[2] == 'TRG' and artf_1.split('_')[2] == 'SRC':
                if artf_2 in trace_dict[artf_1]:
                    rows_list.append({'artf_1': artf_1, 'artf_2':artf_2, 'link': 1})
                else:
                    rows_list.append({'artf_1': artf_1, 'artf_2':artf_2, 'link': 0})
    
    global df_trace
    df_trace = pd.DataFrame(rows_list)


if __name__ == '__main__':
    read_oracle_files()
    dict2df()

    df_trace.to_csv(BASE_DIR + '/oracle/output/trace_matrix.csv', sep=',', index=False)
    df_artifacts_descriptions.to_csv(BASE_DIR + '/oracle/output/artifacts_descriptions.csv', sep=',', index=False)
