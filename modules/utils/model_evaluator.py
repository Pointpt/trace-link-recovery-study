import pandas as pd
import pprint
import pickle
import math

import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix

from sklearn.metrics import precision_score, recall_score, f1_score

from modules.utils import similarity_measures as sm

class ModelEvaluator:
    def __init__(self, oracle):
        self.oracle = oracle
                    
    def get_oracle_df(self):
        return self.oracle
    
    def __fillUp_traceLinksDf(self, model, top_n, sim_threshold):
        trace_links_df = pd.DataFrame(index = model.get_sim_matrix().index,
                                      columns = model.get_sim_matrix().columns,
                                      data = model.get_sim_matrix().values)
        
        for col in trace_links_df.columns:
            nlargest_df = trace_links_df.nlargest(n = top_n, columns=col, keep='first')    
            trace_links_df[col] = [1 if x in nlargest_df[col].tolist() and x >= sim_threshold else 0 for x in trace_links_df[col]]

        return trace_links_df
    
    def evaluate_model(self, verbose=False, file=None, model=None, top_value=None, sim_threshold=None, ref_name=""):        
        trace_links_df = self.__fillUp_traceLinksDf(model=model, top_n=top_value, sim_threshold=sim_threshold)
                       
        eval_df = pd.DataFrame(columns=['precision','recall','fscore'],
                              index=self.oracle.columns)
        eval_df['precision'] = [precision_score(y_true=self.oracle[col],y_pred=trace_links_df[col]) for col in self.oracle.columns]
        eval_df['recall'] = [recall_score(y_true=self.oracle[col],y_pred=trace_links_df[col]) for col in self.oracle.columns]
        eval_df['fscore'] = [f1_score(y_true=self.oracle[col],y_pred=trace_links_df[col]) for col in self.oracle.columns]
        eval_df.index.name = 'Bug_Number'
        eval_df.index = eval_df.index.astype(str)       
        
        mean_precision = eval_df.precision.mean()
        mean_recall = eval_df.recall.mean()
        mean_fscore = eval_df.fscore.mean()
        
        if verbose:
            self.print_report(file)
        
        return {'model':model.get_model_gen_name(), 
                'ref_name':ref_name, 
                'perc_precision': round(mean_precision,4)*100, 
                'perc_recall': round(mean_recall,4)*100,
                'perc_fscore': round(mean_fscore,4)*100,
                'trace_links_df' : trace_links_df, 
                'top':top_value, 
                'sim_threshold':sim_threshold,
                'eval_df':eval_df}
    
    
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
    
        
    def plot_precision_vs_recall(self):
        plt.figure(figsize=(6,6))
        plt.plot(self.eval_df.recall, self.eval_df.precision, 'ro', label='Precision vs Recall')

        plt.ylabel('Precision')
        plt.xlabel('Recall')

        plt.axis([0, 1.1, 0, 1.1])
        plt.title("Precision vs Recall Plot - " + self.model.get_name())
        plt.show()
    
    
    def plot_evaluations_1(self, evals_df, title):
        results = evals_df

        start_pos, width = 0.25, 0.25
        pos_1 = list([start_pos,         start_pos+2,         start_pos+4,         start_pos+6]) 
        pos_2 = list([start_pos+width,   start_pos+2+width,   start_pos+4+width,   start_pos+6+width]) 
        pos_3 = list([start_pos+2*width, start_pos+2+2*width, start_pos+4+2*width, start_pos+6+2*width]) 
        pos_4 = list([start_pos+3*width, start_pos+2+3*width, start_pos+4+3*width, start_pos+6+3*width])                

        f, (ax1,ax2,ax3) = plt.subplots(3,1, figsize=(20,15))
        f.suptitle(title)

        model_names = [m.upper() for m in results.model.unique()]

        ax1.set_title('Percentual Precision')
        ax1.bar(pos_1, width=width, height=results[results.model == 'lsi'].perc_precision.values, color='blue')
        ax1.bar(pos_2, width=width, height=results[results.model == 'lda'].perc_precision.values, color='red')
        ax1.bar(pos_3, width=width, height=results[results.model == 'bm25'].perc_precision.values, color='green')
        ax1.bar(pos_4, width=width, height=results[results.model == 'wordvector'].perc_precision.values, color='orange')
        ax1.set(xlabel='model', ylabel='precision')
        ax1.set_xticks([0.6, 2.6, 4.6, 6.6])
        ax1.set_xticklabels(model_names)
        ax1.set_ylim([0,100])
        ax1.legend(['TOP 1 - COS 0.0', 'TOP 3 - COS 0.0', 'TOP 5 - COS 0.0', 'TOP 10 - COS 0.0'], loc='upper right')
        ax1.grid()

        ax2.set_title('Percentual Recall')
        ax2.bar(pos_1, width=width, height=results[results.model == 'lsi'].perc_recall, color='blue')
        ax2.bar(pos_2, width=width, height=results[results.model == 'lda'].perc_recall, color='red')
        ax2.bar(pos_3, width=width, height=results[results.model == 'bm25'].perc_recall, color='green')
        ax2.bar(pos_4, width=width, height=results[results.model == 'wordvector'].perc_recall, color='orange')
        ax2.set(xlabel='model', ylabel='recall')
        ax2.set_xticks([0.6, 2.6, 4.6, 6.6])
        ax2.set_xticklabels(model_names)
        ax2.set_ylim([0,100])
        ax2.legend(['TOP 1 - COS 0.0', 'TOP 3 - COS 0.0', 'TOP 5 - COS 0.0', 'TOP 10 - COS 0.0'], loc='upper left')
        ax2.grid()

        ax3.set_title('Percentual FScore')
        ax3.bar(pos_1, width=width, height=results[results.model == 'lsi'].perc_fscore, color='blue')
        ax3.bar(pos_2, width=width, height=results[results.model == 'lda'].perc_fscore, color='red')
        ax3.bar(pos_3, width=width, height=results[results.model == 'bm25'].perc_fscore, color='green')
        ax3.bar(pos_4, width=width, height=results[results.model == 'wordvector'].perc_fscore, color='orange')
        ax3.set(xlabel='model', ylabel='fscore')
        ax3.set_xticks([0.6, 2.6, 4.6, 6.6])
        ax3.set_xticklabels(model_names)
        ax3.set_ylim([0,100])
        ax3.legend(['TOP 1 - COS 0.0', 'TOP 3 - COS 0.0', 'TOP 5 - COS 0.0', 'TOP 10 - COS 0.0'], loc='upper right')
        ax3.grid()

    def plot_evaluations_2(self, title, results):
        f,axes = plt.subplots(1,int(len(results)/10),figsize=(25,5))
        f.suptitle(title)
        
        top_values = [1.0, 3.0, 5.0, 10.0, 21.0]
        top_names = ['TOP {}'.format(a) for a in [1,3,5,10,21]]
        
        for i,ax in enumerate(axes):
            results_2 = results[results.top == top_values[i]]
            ax.set_title(top_names[i])
            ax.plot(results_2.sim_threshold, results_2.perc_precision, color='green')
            ax.plot(results_2.sim_threshold, results_2.perc_recall, color='blue')
            ax.plot(results_2.sim_threshold, results_2.perc_fscore, color='red')
            ax.set_ylim([0,100])
            ax.set_xlabel('similarity threshold')
            ax.set_ylabel('metric value (%)')
            ax.legend(['Precision','Recall','FScore'])
            ax.grid()

        #ax2.set_title('TOP 3')
        #ax2.plot(results_t3.sim_threshold, results_t3.perc_precision, color='green')
        #ax2.plot(results_t3.sim_threshold, results_t3.perc_recall, color='blue')
        #ax2.plot(results_t3.sim_threshold, results_t3.perc_fscore, color='red' )
        #ax2.set_ylim([0,100])
        #ax2.set_xlabel('similarity threshold')
        #ax2.set_ylabel('metric value (%)')
        #ax2.legend(['Precision','Recall','FScore'])
        #ax2.grid()
#
        #ax3.set_title('TOP 5')
        #ax3.plot(results_t5.sim_threshold, results_t5.perc_precision, color='green')
        #ax3.plot(results_t5.sim_threshold, results_t5.perc_recall, color='blue')
        #ax3.plot(results_t5.sim_threshold, results_t5.perc_fscore, color='red' )
        #ax3.set_ylim([0,100])
        #ax3.set_xlabel('similarity threshold')
        #ax3.set_ylabel('metric value (%)')
        #ax3.legend(['Precision','Recall','FScore'])
        #ax3.grid()
#
        #ax4.set_title('TOP 10')
        #ax4.plot(results_t10.sim_threshold, results_t10.perc_precision, color='green')
        #ax4.plot(results_t10.sim_threshold, results_t10.perc_recall, color='blue')
        #ax4.plot(results_t10.sim_threshold, results_t10.perc_fscore, color='red' )
        #ax4.set_ylim([0,100])
        #ax4.set_xlabel('similarity threshold')
        #ax4.set_ylabel('metric value (%)')
        #ax4.legend(['Precision','Recall','FScore'])
        #ax4.grid()
        #
        #ax5.set_title('TOP 21')
        #ax5.plot(results_t21.sim_threshold, results_t21.perc_precision, color='green')
        #ax5.plot(results_t21.sim_threshold, results_t21.perc_recall, color='blue')
        #ax5.plot(results_t21.sim_threshold, results_t21.perc_fscore, color='red' )
        #ax5.set_ylim([0,100])
        #ax5.set_xlabel('similarity threshold')
        #ax5.set_ylabel('metric value (%)')
        #ax5.legend(['Precision','Recall','FScore'])
        #ax5.grid()