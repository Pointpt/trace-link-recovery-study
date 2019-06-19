import pandas as pd
import numpy as np
import pprint
import pickle
import math

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from sklearn.metrics import precision_score, recall_score, f1_score

from modules.utils import similarity_measures as sm
from modules.utils import aux_functions

class ModelEvaluator:
    def __init__(self, oracle):
        self.oracle = oracle
                    
    def get_oracle_df(self):
        return self.oracle
    
    def __fillUp_traceLinksDf(self, model, top_n, sim_threshold):
        if 'zero_r' in model.get_model_gen_name(): # zero_r model case
            #print('Zero R Model Detected')
            return model.get_sim_matrix()
        
        trace_links_df = pd.DataFrame(index = model.get_sim_matrix().index,
                                      columns = model.get_sim_matrix().columns,
                                      data = model.get_sim_matrix().values)
        
        for col in trace_links_df.columns:
            nlargest_df = trace_links_df.nlargest(n = top_n, columns=col, keep='first')    
            trace_links_df[col] = [1 if x in nlargest_df[col].tolist() and x >= sim_threshold else 0 for x in trace_links_df[col]]

        return trace_links_df
    
    def evaluate_model(self, verbose=False, file=None, model=None, top_value=None, sim_threshold=None, ref_name=""):        
        trace_links_df = self.__fillUp_traceLinksDf(model=model, top_n=top_value, sim_threshold=sim_threshold)
                       
        #eval_df = pd.DataFrame(columns=['precision','recall','fscore'], index=self.oracle.columns)
        #eval_df['precision'] = [precision_score(y_true=self.oracle[col],y_pred=trace_links_df[col]) for col in self.oracle.columns]
        #eval_df['recall'] = [recall_score(y_true=self.oracle[col],y_pred=trace_links_df[col]) for col in self.oracle.columns]
        #eval_df['fscore'] = [f1_score(y_true=self.oracle[col],y_pred=trace_links_df[col]) for col in self.oracle.columns]
        #eval_df.index.name = 'Bug_Number'
        #eval_df.index = eval_df.index.astype(str)       
        
        #mean_precision = eval_df.precision.mean()
        #mean_recall = eval_df.recall.mean()
        #mean_fscore = eval_df.fscore.mean()
        
        tp = len(aux_functions.get_true_positives( oracle_df = self.oracle, output_df = trace_links_df))
        fp = len(aux_functions.get_false_positives(oracle_df = self.oracle, output_df = trace_links_df))
        fn = len(aux_functions.get_false_negatives(oracle_df = self.oracle, output_df = trace_links_df))
        
        mean_precision = tp / (tp + fp)
        mean_recall = tp / (tp + fn)
        
        # F_2-SCORE
        mean_fscore = None
        if mean_precision == 0 and mean_recall == 0:
            mean_fscore = 0
        else:
            mean_fscore = (1 + 2**2) * (mean_precision * mean_recall) / ((2**2 * mean_precision) + mean_recall) 
        
        if verbose:
            self.print_report(file)
        
        return {'model': model.get_model_gen_name(), 
                'ref_name': ref_name, 
                'perc_precision': round(mean_precision,4)*100, 
                'perc_recall': round(mean_recall,4)*100,
                'perc_fscore': round(mean_fscore,4)*100,
                'trace_links_df' : trace_links_df, 
                'top': top_value, 
                'sim_threshold': sim_threshold}
    
    
    def run_evaluator(self, verbose=False, file=None, models=None, top_values=[1,3,5,10], sim_thresholds=[(sm.SimilarityMeasure.COSINE, 0.0)]):               
        evals = pd.DataFrame(columns=['model','ref_name','perc_precision','perc_recall','perc_fscore'])
        
        for model in models:
            print("Evaluating {} Model ----- ".format(model.get_model_gen_name().upper()))
            for top_value in top_values:
                for s_name,s_threshold in sim_thresholds:
                    ref_name = "top_{}_{}_{}".format(top_value, s_name.value, s_threshold)
                    evals = evals.append(self.evaluate_model(verbose=verbose, 
                                                             model=model,
                                                             top_value=top_value, 
                                                             sim_threshold=s_threshold, 
                                                             ref_name=ref_name), ignore_index=True)
        return evals
    
           
    # plot precision, recall and fscore plot for varied values of tops
    # and similarity threshold fixed in 0.0 for each model
    def plot_evaluations_1(self, evals_df, title):        
        results = evals_df

        start_pos, width = 0.25, 0.25
        pos_1 = list([start_pos,         start_pos+2,         start_pos+4,         start_pos+6]) 
        pos_2 = list([start_pos+width,   start_pos+2+width,   start_pos+4+width,   start_pos+6+width]) 
        pos_3 = list([start_pos+2*width, start_pos+2+2*width, start_pos+4+2*width, start_pos+6+2*width]) 
        #pos_4 = list([start_pos+3*width, start_pos+2+3*width, start_pos+4+3*width, start_pos+6+3*width])                

        f, axes = plt.subplots(3,1, figsize=(24,15))
        f.suptitle(title)

        model_names = [m.upper() for m in results.model.unique()]
        titles = ['Percentual Precision','Percentual Recall','Percentual FScore']
        legends = ['TOP 1 - COS 0.0', 'TOP 3 - COS 0.0', 'TOP 5 - COS 0.0']
        labels = ['precision', 'recall', 'fscore']
        col = ""
        for i,ax in enumerate(axes):
            if i == 0:
                col = 'perc_precision'
            elif i == 1:
                col = 'perc_recall'
            elif i == 2:
                col = 'perc_fscore'
                
            ax.set_title(titles[i])
            ax.bar(pos_1, width=width, height=results[results.model == 'lsi'][col].values, color='black')
            ax.bar(pos_2, width=width, height=results[results.model == 'lda'][col].values, color='darkgray')
            ax.bar(pos_3, width=width, height=results[results.model == 'bm25'][col].values, color='lightgray')
            #ax.bar(pos_4, width=width, height=results[results.model == 'wordvector'][col].values, color='silver')
            ax.set(xlabel='model', ylabel=labels[i])
            ax.set_xticks([0.6, 2.6, 4.6])
            ax.set_xticklabels(model_names)
            ax.set_ylim([0,100])
            ax.legend(legends, loc='upper right')
            ax.grid()
    
    # plot mean precision, recall and fscore of each model
    # based on evaluations made with varied top values and
    # similarity thresholds
    def plot_evaluations_3(self, evals_df, title):
        results = evals_df
        
        start_pos, width = 0.25, 0.25
        pos_1 = list([start_pos,         start_pos+1,         start_pos+2,         start_pos+3           , start_pos+4])    # precisions
        pos_2 = list([start_pos+width,   start_pos+1+width,   start_pos+2+width,   start_pos+3+width     , start_pos+4+width])   # recalls
        pos_3 = list([start_pos+2*width, start_pos+1+2*width, start_pos+2+2*width, start_pos+3+2*width   , start_pos+4+2*width  ])  # fscores      

        positions = [pos_1, pos_2, pos_3]

        f, ax = plt.subplots(1,1, figsize=(10,5))
        f.suptitle(title)

        model_names = [m.upper() for m in results.model.unique()]

        legends = ['Percentual Precision','Percentual Recall','Percentual FScore']

        heights_1 = [np.mean(results[results.model == m.lower()]['perc_precision'].values) for m in model_names]
        heights_2 = [np.mean(results[results.model == m.lower()]['perc_recall'].values) for m in model_names]
        heights_3 = [np.mean(results[results.model == m.lower()]['perc_fscore'].values) for m in model_names]

        labels = [[(pos_1[i], heights_1[i] + .3, str(round(heights_1[i],1)) + '%') for i in range(5)],
                  [(pos_2[i], heights_2[i] + .3, str(round(heights_2[i],1)) + '%') for i in range(5)],
                  [(pos_3[i], heights_3[i] + .3, str(round(heights_3[i],1)) + '%') for i in range(5)]]
        
        ax.bar(pos_1, width=width, height=heights_1, color='black')
        ax.bar(pos_2, width=width, height=heights_2, color='darkgray')
        ax.bar(pos_3, width=width, height=heights_3, color='lightgray')

        for l in labels:
            for x,y,label in l:
                ax.text(x=x, y=y, s=label, ha='center', va='bottom', color='black')
        
        ax.set(xlabel='Model', ylabel='Mean Metric Value (%)')
        ax.set_xticks([0.5, 1.5, 2.5, 3.5, 4.5])
        ax.set_xticklabels(model_names)
        ax.set_ylim([0,100])
        ax.legend(legends, loc='upper right')
        
    
    # plot precision, recall and fscore for a single model, varying
    # similarity thresholds range(0.0, 0.9) and the top values (1,3,5)
    def plot_evaluations_2(self, title, results, output_file="", model_name=""):        
        f,axes = plt.subplots(1,3,figsize=(25,5))
        f.suptitle(title)

        top_values = [1.0, 3.0, 5.0]
        top_names = ['TOP {}'.format(a) for a in [1,3,5]]

        for i,ax in enumerate(axes):
            results_2 = results[(results.top == top_values[i]) & (results.model == model_name)]
            ax.set_title(top_names[i])
            ax.plot(results_2.sim_threshold, results_2.perc_precision, marker='o', linestyle='dashed', color='blue')
            ax.plot(results_2.sim_threshold, results_2.perc_recall, marker='v', linestyle='dashed', color='green')
            ax.plot(results_2.sim_threshold, results_2.perc_fscore, marker='^', linestyle='dashed', color='brown')
            ax.hlines(y=42.85, color='red', xmin=0, xmax=1)
            ax.hlines(y=55.26, color='red', xmin=0, xmax=1)
            ax.hlines(y=66.66, color='red', xmin=0, xmax=1)
            ax.set_ylim([0,100])
            ax.set_xlabel('similarity threshold')
            ax.set_ylabel('metric value (%)')
            ax.legend(['Precision','Recall','F2-Score'])
            ax.grid()
        
        if output_file != "":
            path = '/home/guilherme/Dropbox/Aplicativos/Overleaf/ESEM 2019 Paper/imgs/'
            plt.savefig(path + output_file + '.eps', format='eps', bbox_inches='tight', dpi=1200, pad_inches=.3)
    
    # plot precision, recall for a single model, varying
    # similarity thresholds range(0.0, 0.9) and the top values (10,20,40) (BR_TC Context)
    def plot_evaluations_2_2(self, title, results, model_name):        
        f,axes = plt.subplots(1,3,figsize=(25,5))
        f.suptitle(title)

        top_values = [10.0, 20.0, 40.0]
        top_names = ['TOP {}'.format(a) for a in [10,20,40]]

        for i,ax in enumerate(axes):
            results_2 = results[(results.top == top_values[i]) & (results.model == model_name)]
            ax.set_title(top_names[i])
            ax.plot(results_2.sim_threshold, results_2.perc_precision, marker='o', linestyle='dashed', color='blue')
            ax.plot(results_2.sim_threshold, results_2.perc_recall, marker='v', linestyle='dashed', color='green')
            ax.plot(results_2.sim_threshold, results_2.perc_fscore, marker='^', linestyle='dashed', color='brown')
            ax.hlines(y=42.85, color='red', xmin=0, xmax=1)
            ax.set_ylim([0,100])
            ax.set_xlabel('similarity threshold')
            ax.set_ylabel('metric value (%)')
            ax.legend(['Precision','Recall','F2-Score'])
            ax.grid()
    
    # plot precision x recall graphs for each model in a figure with 4 axes
    def plot_evaluations_4(self, results):
        f,axes = plt.subplots(1,4, figsize=(20,5))
        models_names = ['lsi', 'lda', 'bm25', 'wordvector']
        line_styles = ['go--', 'bo--', 'ro--', 'k+--']
        
        for i,ax in enumerate(axes):
            results_subset = results[results.model == models_names[i]]
            results_subset.sort_values('perc_recall', inplace=True)
            ax.plot(results_subset.perc_recall, results_subset.perc_precision, line_styles[i], linewidth=1.5)
            ax.set_xlabel('recall')
            ax.set_ylabel('precision')
            ax.set_title(models_names[i].upper() + ' Evaluation')
            ax.set_ylim(0,100)
            ax.set_xlim(0,100)

    # plot precision x recall graph in a single figure for all the models
    def plot_evaluations_5(self, results):
        f,ax = plt.subplots(1,1, figsize=(10,5))
        models_names = ['lsi', 'lda', 'bm25', 'wordvector']
        #line_styles = ['v--', '^--', 'o--', '>--']
        #colors = ['black', 'black', 'black', 'black']
        line_styles = ['g^--', 'yv--', 'bo--', 'k+--']

        for i in range(4):
            results_subset = results[results.model == models_names[i]]
            results_subset.sort_values('perc_recall', inplace=True)
            ax.plot(results_subset.perc_recall, results_subset.perc_precision, line_styles[i], linewidth=1.5)
            ax.set_xlabel('recall')
            ax.set_ylabel('precision')
            #ax.set_title('All Techniques Evaluation')
            ax.set_ylim(0,100)
        
        zeror_precision = results[results.model == 'zero_r'].perc_precision
        zeror_recall = results[results.model == 'zero_r'].perc_recall
        
        ax.hlines(y=zeror_precision, color='red', xmin=0, xmax=zeror_recall)
        ax.vlines(x=zeror_recall, color='red', ymin=0, ymax=zeror_precision)
        ax.legend(['LSI','LDA','BM25','Word Vector', 'Zero R'])