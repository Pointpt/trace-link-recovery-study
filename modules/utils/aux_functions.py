from itertools import product
import pickle
import seaborn as sns
import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt
from matplotlib_venn import venn3

from PIL import Image
from wordcloud import WordCloud
from collections import Counter

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
        
def highlight_df(df, palete="green"):
    cm = sns.light_palette(palete, as_cmap=True)
    return df.style.background_gradient(cmap=cm) 


def calculate_sparsity(X):
    non_zero = np.count_nonzero(X)
    total_val = np.product(X.shape)
    sparsity = (total_val - non_zero) / total_val
    return sparsity


# get true positives list compared to an oracle
def get_true_positives(oracle_df, output_df):
    true_positives = []
    for idx,row in output_df.iterrows():
        for col in output_df.columns:
            if row[col] == 1 and oracle_df.at[idx, col] == 1:
                true_positives.append((idx,col))
    return set(true_positives)

# get false positives list compared to an oracle
def get_false_positives(oracle_df, output_df):
    false_positives = [] 
    for idx,row in output_df.iterrows():
        for col in output_df.columns:
            if row[col] == 1 and oracle_df.at[idx, col] == 0:
                false_positives.append((idx,col))
    return set(false_positives)

# get false negative list compared to an oracle
def get_false_negatives(oracle_df, output_df):
    false_negatives = []
    for idx,row in output_df.iterrows():
        for col in output_df.columns:
            if row[col] == 0 and oracle_df.at[idx, col] == 1:
                false_negatives.append((idx,col))
    return set(false_negatives)

# function to reuturn the trace_link_matrix associated with a specified precision value or recall value
# for a specific model
def get_trace_links_df(evaluations_df, model, perc_precision="", perc_recall=""):
    if perc_precision != "":
        df = evaluations_df[(evaluations_df.model == model) & (evaluations_df.perc_precision == perc_precision)]
        return df.iloc[-1,:].trace_links_df
    elif perc_recall != "":
        df = evaluations_df[(evaluations_df.model == model) & (evaluations_df.perc_recall == perc_recall)]
        return df.iloc[-1,:].trace_links_df


# function to create word clouds with the tokens of features and bug reports
# associated with the results of a specific model
def create_wordcloud_feat_br(model_exc_set, bugreports, features, wc_feat_title, wc_br_title):
    feat_texts = ""
    brs_texts = ""
    for feat,br in model_exc_set:
        aux_text_1 = " ".join(features[features.Feature_Shortname == feat].tokens.values[0])
        aux_text_2 = " ".join(bugreports[bugreports.Bug_Number == br].tokens.values[0])
        feat_texts = feat_texts + " " + aux_text_1
        brs_texts = brs_texts + " " + aux_text_2
    
    wordcloud_feats = WordCloud(max_font_size=50, max_words=20, background_color="white").generate(feat_texts)
    plt.figure()
    plt.imshow(wordcloud_feats, interpolation="bilinear")
    plt.axis("off")
    plt.title(wc_feat_title)
    plt.show()
    
    wordcloud_brs = WordCloud(max_font_size=50, max_words=20, background_color="white").generate(brs_texts)
    plt.figure()
    plt.imshow(wordcloud_brs, interpolation="bilinear")
    plt.axis("off")
    plt.title(wc_br_title)
    plt.show()

    
# function to create word clouds with the tokens of test cases and bug reports
# associated with the results of a specific model
def create_wordcloud_tc_br(model_exc_set, bugreports, testcases, wc_tc_title, wc_br_title):
    tcs_texts = ""
    brs_texts = ""
    for tc,br in model_exc_set:
        aux_text_1 = " ".join(testcases[testcases.tc_name == tc].tokens.values[0])
        aux_text_2 = " ".join(bugreports[bugreports.br_name == br].tokens.values[0])
        tcs_texts = tcs_texts + " " + aux_text_1
        brs_texts = brs_texts + " " + aux_text_2
    
    wordcloud_feats = WordCloud(max_font_size=50, max_words=20, background_color="white").generate(tcs_texts)
    plt.figure()
    plt.imshow(wordcloud_feats, interpolation="bilinear")
    plt.axis("off")
    plt.title(wc_tc_title)
    plt.show()
    
    wordcloud_brs = WordCloud(max_font_size=50, max_words=20, background_color="white").generate(brs_texts)
    plt.figure()
    plt.imshow(wordcloud_brs, interpolation="bilinear")
    plt.axis("off")
    plt.title(wc_br_title)
    plt.show()
    
# function to detail the features of testcases and the summary of bug reports
# associated with a exclusive set of results of a specific model
def detail_features_tc_br(exc_set, testcases, bugreports):
    pd.set_option('max_colwidth', 400)
    df = pd.DataFrame(columns=['tc','tc_feat','br','br_summary'])
    df['tc'] = [tc for tc,br in exc_set]
    df['tc_feat'] = [testcases[testcases.TC_Number == tc].Firefox_Feature.values[0] for tc,br in exc_set]
    df['br'] = [br for tc,br in exc_set]
    df['br_summary'] = [bugreports[bugreports.Bug_Number == br].Summary.values[0] for tc,br in exc_set]
    #display(df)
    return df


# summarizes the grouped traces for each tuple bug report,feature to a list of related test cases
def detail_features_tc_br_groups(tc_br_list, testcases, bugreports):
    grouped = detail_features_tc_br(tc_br_list, testcases=testcases, bugreports=bugreports).groupby(['br','br_summary','tc_feat']).groups
    df = pd.DataFrame(columns=['br','br_summary','tc_feat','tc_ids'])
    l = list(tc_br_list)
    for idx,(key,val) in enumerate(grouped.items()):
        df.at[idx,'br'] = key[0]
        df.at[idx,'br_summary'] = key[1]
        df.at[idx,'tc_feat'] = key[2]
        df.at[idx,'tc_ids'] = [l[i][0] for i in val]
    return df

# function to details the features related to bug reports and the summary of these bug reports
# associated with a exclusive set of results of a specific model
def detail_features_br(exc_set, features, bugreports):
    pd.set_option('max_colwidth', 400)
    df = pd.DataFrame(columns=['feat','feat_desc','br','br_summary'])
    df['feat'] = [feat for feat,br in exc_set]
    df['feat_desc'] = [features[features.Feature_Shortname == feat].Feature_Description.values[0] for feat,br in exc_set]
    df['br'] = [br for feat,br in exc_set]
    df['br_summary'] = [bugreports[bugreports.Bug_Number == br].Summary.values[0] for feat,br in exc_set]
    #display(df)
    return df
    

def get_retrieved_traces_df(oracle, evals_df, top_values, sim_threshs):
    MODELS = ['lsi','lda','bm25','wordvector']
    TOP_VALUES = top_values
    SIM_THRESHOLDS = sim_threshs
    
    retrieved_traces_df = pd.DataFrame(columns=['top','sim_thresh','model','retrieved','num_TP','num_FP','num_FN','TP','FP','FN','precision','recall'])
       
    for m in MODELS:
        for top in TOP_VALUES:
            for sim_thresh in SIM_THRESHOLDS:
                df = evals_df[(evals_df.ref_name == 'top_{}_cosine_{}'.format(top,sim_thresh)) & (evals_df.model == m)].iloc[0,:]
                trace_links = df.trace_links_df

                tp = get_true_positives( oracle_df=oracle, output_df=trace_links)
                fp = get_false_positives(oracle_df=oracle, output_df=trace_links)
                fn = get_false_negatives(oracle_df=oracle, output_df=trace_links)

                ans = {'top': top, 
                       'sim_thresh': sim_thresh, 
                       'model': m,
                       'retrieved': sum([trace_links[col].sum() for col in trace_links.columns]),
                       'precision': df.perc_precision,
                       'recall': df.perc_recall,
                       'fscore': df.perc_fscore,
                       'TP': tp,
                       'num_TP': len(tp),
                       'FP': fp,
                       'num_FP': len(fp),
                       'FN': fn,
                       'num_FN': len(fn)} 

                retrieved_traces_df = retrieved_traces_df.append(ans, ignore_index=True)
    
    retrieved_traces_df.sort_values(by='top', inplace=True)
    return retrieved_traces_df


def get_oracle_true_positives(strat_runner):
    oracle_true_traces = set()
    oracle = strat_runner.get_oracle()

    for idx,row in oracle.iterrows():
        for col in oracle.columns:
            if oracle.at[idx, col] == 1:
                oracle_true_traces.add((idx,col))
    
    return oracle_true_traces


def get_captured_traces_union(top_value, retrieved_traces_df):
    bm25_true_traces = retrieved_traces_df[(retrieved_traces_df.model == 'bm25') & (retrieved_traces_df.top == top_value)].iloc[0,:].TP
    lsi_true_traces = retrieved_traces_df[(retrieved_traces_df.model == 'lsi') & (retrieved_traces_df.top == top_value)].iloc[0,:].TP
    lda_true_traces = retrieved_traces_df[(retrieved_traces_df.model == 'lda') & (retrieved_traces_df.top == top_value)].iloc[0,:].TP
    wordvector_true_traces = retrieved_traces_df[(retrieved_traces_df.model == 'wordvector') & (retrieved_traces_df.top == top_value)].iloc[0,:].TP
    
    return(bm25_true_traces | lsi_true_traces | lda_true_traces | wordvector_true_traces)


def get_captured_traces_intersec(top_value, retrieved_traces_df):
    bm25_true_traces = retrieved_traces_df[(retrieved_traces_df.model == 'bm25') & (retrieved_traces_df.top == top_value)].iloc[0,:].TP
    lsi_true_traces = retrieved_traces_df[(retrieved_traces_df.model == 'lsi') & (retrieved_traces_df.top == top_value)].iloc[0,:].TP
    lda_true_traces = retrieved_traces_df[(retrieved_traces_df.model == 'lda') & (retrieved_traces_df.top == top_value)].iloc[0,:].TP
    wordvector_true_traces = retrieved_traces_df[(retrieved_traces_df.model == 'wordvector') & (retrieved_traces_df.top == top_value)].iloc[0,:].TP
    
    return(bm25_true_traces & lsi_true_traces & lda_true_traces & wordvector_true_traces)


def get_traces_set(retrieved_traces, top_value, traces_type):    
    bm25_set = retrieved_traces[(retrieved_traces.model == 'bm25')       & (retrieved_traces.top == top_value)].iloc[0,:][traces_type]
    lsi_set  = retrieved_traces[(retrieved_traces.model == 'lsi')        & (retrieved_traces.top == top_value)].iloc[0,:][traces_type]
    lda_set  = retrieved_traces[(retrieved_traces.model == 'lda')        & (retrieved_traces.top == top_value)].iloc[0,:][traces_type]
    wv_set   = retrieved_traces[(retrieved_traces.model == 'wordvector') & (retrieved_traces.top == top_value)].iloc[0,:][traces_type]
    return (bm25_set, lsi_set, lda_set, wv_set)


def plot_venn_diagrams(top_value, bm25_set, lsi_set, lda_set, wv_set, traces_type):
    venn3([bm25_set, lsi_set, lda_set], ['BM25','LSI','LDA'])
    plt.title('Comparison {} by Model (BM25, LSI, LDA) - TOP {}'.format(traces_type, top_value))
    plt.show()

    venn3([bm25_set, wv_set, lda_set], ['BM25','WV','LDA'])
    plt.title('Comparison {} by Model (BM25, WV, LDA) - TOP {}'.format(traces_type, top_value))
    plt.show()

    venn3([lsi_set, wv_set, lda_set], ['LSI','WV','LDA'])
    plt.title('Comparison {} by Model (LSI, WV, LDA) - TOP {}'.format(traces_type, top_value))
    plt.show()

    venn3([lsi_set, wv_set, bm25_set], ['LSI','WV','BM25'])
    plt.title('Comparison {} by Model (LSI, WV, BM25) - TOP {}'.format(traces_type, top_value))
    plt.show()

    
def get_exclusive_traces(bm25_set, lsi_set, lda_set, wv_set, traces_type, verbose=False):
    print("BM25 Exclusive {}:".format(traces_type))
    bm25_exc_set = bm25_set - lsi_set - lda_set - wv_set
    if verbose: 
        display(bm25_exc_set)
    print("len(bm25_exc_set): {}".format(len(bm25_exc_set)))

    print("\n\nLSI Exclusive {}:".format(traces_type))
    lsi_exc_set = lsi_set - bm25_set - lda_set - wv_set
    if verbose:
        display(lsi_exc_set)
    print("len(lsi_exc_set): {}".format(len(lsi_exc_set)))

    print("\n\nLDA Exclusive {}:".format(traces_type))
    lda_exc_set = lda_set - lsi_set - bm25_set - wv_set
    if verbose:
        display(lda_exc_set)
    print("len(lda_exc_set): {}".format(len(lda_exc_set)))

    print("\n\nWV Exclusive {}:".format(traces_type))
    wv_exc_set = wv_set - lda_set - lsi_set - bm25_set
    if verbose:
        display(wv_exc_set)
    print("len(wv_exc_set): {}".format(len(wv_exc_set)))
    
    return (bm25_exc_set, lsi_exc_set, lda_exc_set, wv_exc_set)


def _calc_goodness(row):
    precision = row['precision']
    recall = row['recall']
    if precision >= 20 and recall >= 60:
        return "Acceptable"
    elif precision >= 30 and recall >= 70:
        return "Good"
    elif precision >= 50 and recall >= 80:
        return "Excellent"
    else:
        return "-"

def calculate_goodness(evals):
    model_names = ['bm25','lsi','lda','wordvector']
    
    df = pd.DataFrame(columns=['model','precision','recall','goodness'])
    df.model = model_names
    df.precision = [round(np.mean(evals[evals.model == m.lower()]['perc_precision'].values), 2) for m in model_names]
    df.recall =    [round(np.mean(evals[evals.model == m.lower()]['perc_recall'].values), 2) for m in model_names]
    df.goodness = df.apply(lambda row : _calc_goodness(row), axis=1)
    
    return df


## rank the top values of ranking for a given similarity matrix or
## oracle and a selected list of bug reports (brs_list)
def rank_tc_br_matrix(brs_list, matrix, top_value):
    tcs_set = set()
    for br in brs_list:
        tcs_list = matrix.nlargest(n=top_value, columns=br, keep='first').index.to_list()
        for tc in tcs_list:
            tcs_set.add(tc)

    return matrix.loc[tcs_set,brs_list].sort_index()


# return most common words for documents in list of traces
# (source and target artifacts)
def get_mrw_traces_set(traces_list, model):
    df = pd.DataFrame(columns=['trg_artf','src_artf','mrw_trg_artf','mrw_src_artf','trg_artf_dl','src_artf_dl'])
    for idx,(d1_num,d2_num) in enumerate(traces_list):    
        df.at[idx,'trg_artf'] = d1_num
        df.at[idx,'src_artf'] = d2_num
        df.at[idx,'mrw_trg_artf'] = model.docs_feats_df.loc[d1_num,'mrw']
        df.at[idx,'mrw_src_artf'] = model.docs_feats_df.loc[d2_num,'mrw']
        df.at[idx,'trg_artf_dl'] =  model.docs_feats_df.loc[d1_num,'dl']
        df.at[idx,'src_artf_dl'] =  model.docs_feats_df.loc[d2_num,'dl']
    
    word_list_trg_artf = [w for l in df.mrw_trg_artf.values for w in l]
    most_common_words_trg_artfs = Counter(word_list_trg_artf).most_common()[0:6]
    
    word_list_src_artf = [w for l in df.mrw_src_artf.values for w in l]
    most_common_words_src_artfs = Counter(word_list_src_artf).most_common()[0:6]
    
    return(most_common_words_trg_artfs, most_common_words_src_artfs)


