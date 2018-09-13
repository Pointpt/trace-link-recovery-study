
import pandas as pd

files = ['CC_CC', 'CC_TC', 
            'ID_TC', 'ID_ID', 'ID_CC', 'ID_UC',
            'TC_CC', 'TC_TC', 
            'UC_CC', 'UC_ID', 'UC_TC', 'UC_UC']

trace_dict = {}
df_trace = None

sources_names = []
targets_names = []

def read_oracle_files():
    path = 'data/EasyClinic/EasyClinicDataset/oracle/'
    for f_name in files:
        fl = open(path + f_name + '.txt')
        lines = fl.readlines()
        source_abv = f_name.split('_')[0] + '_'
        target_abv = f_name.split('_')[1] + '_'
        for l in lines:
            source, targets = l.split(':')[0], l.split(':')[1]
            targets_list = targets.split(' ')
            
            s_name = source_abv + source.replace('.txt', '') + '_SRC'
            t_list_names = [target_abv  + t.replace('.txt', '') + '_TRG' for t in targets_list[:-1]]

            trace_dict[s_name] = t_list_names

            if s_name not in sources_names:
                sources_names.append(s_name)
            
            for t_name in t_list_names:
                if t_name not in targets_names:
                    targets_names.append(t_name)


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
    encode_artifacts()
    dict2df()

    df_trace.to_csv('data/EasyClinic/EasyClinicDataset/oracle/output/trace_matrix.csv', sep=',', index=False)

