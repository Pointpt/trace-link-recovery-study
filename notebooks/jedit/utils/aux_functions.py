from itertools import product
import pickle
import seaborn as sns

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

      
def report_best_model(results_df):
    print("------------ Report -------------------\n")
    print("Total of Analyzed Hyperparameters Combinations: {}".format(results_df.shape[0]))

    print("\nBest Model Hyperparameters Combination Found:\n")            

    row_idx = results_df['model_dump'][results_df.recall == results_df.recall.max()].index[0]
    bm = pickle.load(open(results_df['model_dump'][row_idx], 'rb'))
    evalu = pickle.load(open(results_df['evaluator_dump'][row_idx], 'rb'))
    evalu.evaluate_model(verbose=True)

    #print("\nPlot Precision vs Recall - Best Model")
    #evalu.plot_precision_vs_recall()

    #print("\nHeatmap of All Models")
    #plot_heatmap(results_df)

    #evalu.save_log()
    
    return bm
    
def print_report_top_3_and_5_v1(results_df):
    for top in [3,5]:
        row_idx_top = results_df[results_df.top_value == top].recall.argmax()
        bm_top = pickle.load(open(results_df['model_dump'][row_idx_top], 'rb'))
        evalu_top = pickle.load(open(results_df['evaluator_dump'][row_idx_top], 'rb'))
        evalu_top.evaluate_model(verbose=True)
        print("------------------------------------------------------------------")

def print_report_top_3_and_5_v2(results_df, metric_threshold, metric):
    for top in [3,5]:
        row_idx_top = results_df[(results_df.top_value == top) & (results_df.metric_value == metric_threshold) & (results_df.metric == metric)].recall.argmax()
        bm_top = pickle.load(open(results_df['model_dump'][row_idx_top], 'rb'))
        evalu_top = pickle.load(open(results_df['evaluator_dump'][row_idx_top], 'rb'))
        evalu_top.evaluate_model(verbose=True)
        print("------------------------------------------------------------------")


def print_report_top_3_and_5_v3(results_df, metric):
    for top in [3,5]:
        row_idx_top = results_df[(results_df.top_value == top) & (results_df.metric == metric)].recall.argmax()
        bm_top = pickle.load(open(results_df['model_dump'][row_idx_top], 'rb'))
        evalu_top = pickle.load(open(results_df['evaluator_dump'][row_idx_top], 'rb'))
        evalu_top.evaluate_model(verbose=True)
        print("------------------------------------------------------------------")
        
def highlight_df(df):
    cm = sns.light_palette("green", as_cmap=True)
    return df.style.background_gradient(cmap=cm)   