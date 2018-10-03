
import pandas as pd
import codecs

files = ['BR_UC']

trace_dict = {}
df_trace = None

sources_names = []
targets_names = []

df_artifacts_descriptions = pd.DataFrame(columns=['artf_name','artf_description'])

BASE_DIR = 'data/jEdit/jEditDataset/'


def _register_source_artifact(s_name):
    if s_name not in sources_names:
        sources_names.append(s_name)
    
    print("s_name: " + s_name)

    global df_artifacts_descriptions

    if s_name not in list(df_artifacts_descriptions.artf_name):
        new_row = []
        
        artf_type = s_name.split('_')[0]   # BR, UC
        artf_num = s_name.split('_')[1]    # 1, 2, 3, ...

        with codecs.open('{0}{1}/{2}/{3}.txt'.format(BASE_DIR, 'docs_english', artf_type, artf_num), encoding='utf-8') as fl:
            artf_desc = fl.read()
            d = {'artf_name' : s_name, 'artf_description' : artf_desc}

            new_row.append(d)
    
            df_artifacts_descriptions = df_artifacts_descriptions.append(new_row, sort=False)


def _register_target_artifact(t_list_names):
    print("t_list_names: " + str(t_list_names))

    for t_name in t_list_names:
        if t_name not in targets_names:
            targets_names.append(t_name)

            print("t_name: " + t_name)

            global df_artifacts_descriptions

            if t_name not in list(df_artifacts_descriptions.artf_name):
                new_row = []
                
                artf_type = t_name.split('_')[0]   # UC, CC, ID, TC
                artf_num = t_name.split('_')[1]    # 1, 2, 3, ...

                with codecs.open('{0}{1}/{2}/{3}.txt'.format(BASE_DIR, 'docs_english', artf_type, artf_num), encoding='utf-8') as fl:
                    artf_desc = fl.read()
                    d = {'artf_name' : t_name, 'artf_description' : artf_desc}

                    new_row.append(d)
            
                    df_artifacts_descriptions = df_artifacts_descriptions.append(new_row, sort=False)


def read_oracle_files():
    for f_name in files:
        with open(BASE_DIR + 'oracle/' + f_name + '.txt', 'r') as fl:
            lines = fl.readlines()
            source_abv = f_name.split('_')[0] + '_'
            target_abv = f_name.split('_')[1] + '_'
            for l in lines:
                source, targets = l.split(':')[0], l.split(':')[1]
                targets_list = targets.split(' ')
                
                s_name = source_abv + source.replace('.txt', '') + '_SRC'
                t_list_names = [target_abv  + t.replace('.txt', '').replace('\n','') + '_TRG' for t in targets_list]

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
