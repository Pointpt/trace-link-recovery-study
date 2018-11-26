from itertools import product

def generate_params_comb_list(**kwargs):
    list_params = []
    for key, values in kwargs.items():
        aux_list = []
        for v in values:
            aux_list.append((key, v))
        list_params.append(aux_list)
    
    list_tuples = list(product(*list_params))
    
    list_dicts = []
    for ex_tup in list_tuples:
        dic = {}
        for in_tup in ex_tup:
            dic[in_tup[0]] = in_tup[1]
        list_dicts.append(dic)
        
    return list_dicts


def plot_heatmap(results_df):
    tmp_df = pd.DataFrame({'precision': results_df['precision'], 
                           'recall' : results_df['recall'], 
                           'fscore': results_df['fscore'], 
                           'model': results_df['model_name']})
    tmp_df.set_index('model', inplace=True)
    fig, ax = plt.subplots(figsize=(10, 4 * 10)) 
    ax = sns.heatmap(tmp_df, vmin=0, vmax=1, linewidths=.5, cmap="Greens", annot=True, cbar=False, ax=ax)


def highlight_df(df):
    cm = sns.light_palette("green", as_cmap=True)
    return df.style.background_gradient(cmap=cm)   


