from itertools import product
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import cohen_kappa_score

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


def calculate_sparsity(X):
    non_zero = np.count_nonzero(X)
    total_val = np.product(X.shape)
    sparsity = (total_val - non_zero) / total_val
    return sparsity


def compile_results(results_dict):
    results = pd.DataFrame(columns=['model','precision','recall','fscore'])
    
    results['model'] = [results_dict['lsi_model'].get_name(), 
                        results_dict['lda_model'].get_name(), 
                        results_dict['bm25_model'].get_name(), 
                        results_dict['w2v_model'].get_name()]

    results['precision'] = [results_dict['lsi_eval'].get_mean_precision(), 
                            results_dict['lda_eval'].get_mean_precision(), 
                            results_dict['bm25_eval'].get_mean_precision(), 
                            results_dict['w2v_eval'].get_mean_precision()]

    results['recall'] = [results_dict['lsi_eval'].get_mean_recall(),
                         results_dict['lda_eval'].get_mean_recall(),
                         results_dict['bm25_eval'].get_mean_recall(),
                         results_dict['w2v_eval'].get_mean_recall()]

    results['fscore'] = [results_dict['lsi_eval'].get_mean_fscore(),
                         results_dict['lda_eval'].get_mean_fscore(),
                         results_dict['bm25_eval'].get_mean_fscore(),
                         results_dict['w2v_eval'].get_mean_fscore()]

    results['precision_perc'] = results.precision.apply(lambda x : 100 * x)
    results['recall_perc'] = results.recall.apply(lambda x : 100 * x)
    results['fscore_perc'] = results.fscore.apply(lambda x : 100 * x)
    
    return results


def plot_results(results_df, title):
    f, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(20,5))
    f.suptitle(title)

    model_names = [m.split('_')[0] for m in results_df.model.values]

    ax1.set_title('Percentual Precision')
    ax1.bar(model_names, results_df.precision_perc, color='blue')
    ax1.set(xlabel='model', ylabel='precision')

    ax2.set_title('Percentual Recall')
    ax2.bar(model_names, results_df.recall_perc, color='red')
    ax2.set(xlabel='model', ylabel='recall')

    ax3.set_title('Percentual FScore')
    ax3.bar(model_names, results_df.fscore_perc, color='green')
    ax3.set(xlabel='model', ylabel='fscore')


def shift_taskruns_answers(taskruns):
    new_answers = list(taskruns.answers.values)
    new_answers = [new_answers[-1]] + new_answers
    del new_answers[-1]
    taskruns['new_answers'] = new_answers
    return taskruns