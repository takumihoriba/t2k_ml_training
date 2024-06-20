# implement later
from analyze_output.analyze_ml_regression import analyze_ml_regression
from analyze_output.analyze_regression import analysisResults

import numpy as np

# import json
# import csv

import pandas as pd

import matplotlib.pyplot as plt
# import seaborn as sns

class MultiRegressionAnalysis:
    def __init__(self, settings, sub_dir_names, percents):
        self.analysis_dict = {}
        self.settings = settings
        self.base_path = settings.mlPath
        self.percents = percents
        self.sub_dirs = sub_dir_names
        self.plot_counter = 0
        self.computed = False

        self.bias_raw_df = None
        self.bias_summary_df = None

        # initialize analysis dictionary
        for p in list(set(percents)):
            self.analysis_dict[p] = []

        print('constructor done')
    
    def analyze(self, tasks=['all']):
        self.compute_bias_summary_stats()
        self.plot_errorbars()


    def compute_bias_summary_stats(self):
        settings = self.settings
        percents = self.percents
        sub_dir_names = self.sub_dirs

        BASE_PATH = settings.mlPath
        df = pd.DataFrame(columns=['dead PMT rate', 'var', 'quantile', 'quantile error', 'median', 'median error'])

        for i, sub_dir in enumerate(sub_dir_names):
            print('BASE_PATH', BASE_PATH)
            print('particleLabel', settings.particleLabel)
            print('inputPath', settings.inputPath)
            print('fitqunPath', settings.fitqunPath)

            print('target', settings.target)
            settings.mlPath = BASE_PATH + sub_dir + '/'
            print('mlPath', settings.mlPath)
            single_ml_analysis, multi_ml_analysis = analyze_ml_regression(settings) 
            # print(f"SINGLE ANALYSIS: {single_ml_analysis}")
            # print(f"MULTI ANALYSIS: {multi_ml_analysis}")
            # ar.add_global_perf("ML", single_ml_analysis[0], single_ml_analysis[1], single_ml_analysis[2], single_ml_analysis[3], single_ml_analysis[4])
            # ar.add_var_perf("ML",multi_ml_analysis) # for now, only single analysis

            # self.analysis_dict[percents[i]].append(ar)

            for j in range(len(single_ml_analysis[0])):
                new_row = {'dead PMT rate': percents[i], 'var': single_ml_analysis[0][j], 'quantile': single_ml_analysis[1][j], 'quantile error': single_ml_analysis[2][j], 'median': single_ml_analysis[3][j], 'median error': single_ml_analysis[4][j]}
                df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        
        df_summary = df.groupby(by=['dead PMT rate', 'var']).describe()
        print('summary stats', df_summary)

        df.to_csv(settings.outputPlotPath + 'reg_analysis_metrics.csv', sep=',', index=False)
        df_summary.to_csv(settings.outputPlotPath + 'reg_analysis_metrics_summary_stats.csv', sep=',', index=False)

        self.bias_raw_df = df
        self.bias_summary_df = df_summary

        self.computed = True

    def plot_errorbars(self, file_path=None):
        if file_path is not None and self.computed == False:
            df = pd.read_csv(file_path)
            self.bias_raw_df = df
        elif self.computed == False:
            self.compute_bias_summary_stats()
        
        df = self.bias_raw_df

        color_choices=[
            '#1f77b4', '#ff7f0e', '#2ca02c', 
            '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
            '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
            '#17becf', '#9edae5']
        
        for i, var in enumerate(df['var'].unique()):
            for metric in ['quantile', 'median']:
                dfsem = df[df['var'] == var].groupby(by=['dead PMT rate', 'var']).sem(ddof=0).reset_index()
                dfmean = df[df['var'] == var].groupby(by=['dead PMT rate', 'var']).mean().reset_index()
                
                fig, ax = plt.subplots()

                # Replace NaN values 
                for r in range(len(dfsem)):
                    if dfsem.iloc[r].isnull().values.any():
                        # if it does, print the row
                        # print(dfsem.iloc[r])
                        # substitude the row with values from original df. For quantile, we use quantile error from original df
                        dfsem.loc[r, 'quantile'] = df[df['var'] == var]['quantile error'].loc[r]
                        dfsem.loc[r, 'median'] = df[df['var'] == var]['median error'].loc[r]

                plt.errorbar(dfmean['dead PMT rate'], y=dfmean[metric], yerr=dfsem[metric], fmt='o', color = color_choices[i], capsize=5)

                ax.set_title(f'Mean and Std of {metric[0].capitalize() + metric[1:]}s of (True - Prediction)s for {var} Axis')
                ax.set_xlabel('Dead PMT Rate [%]')
                ax.set_ylabel(metric[0].capitalize() + metric[1:] + ' [cm]')
                ax.grid(True, linestyle='--', linewidth=0.5)
                fig.savefig(self.settings.outputPlotPath + f'mean_{var}_axis_{metric}.png')

                plt.close(fig)



        # for i, var in enumerate(df['var'].unique()):
        #     for metric in ['quantile', 'median']:

        #         fig, ax = plt.subplots()

        #         df_x_g = df[df['var'] == var].groupby(by=['dead PMT rate', 'var']).describe().reset_index()
        #         df_x_g[0, (metric,'std')] = df[f'{metric} error'][0]

        #         file_path = self.settings.outputPlotPath
        #         ax.errorbar(x=df_x_g['dead PMT rate'], y=df_x_g[metric]['mean'], 
        #                     yerr=df_x_g[metric]['std'],
        #                     fmt='o',
        #                     color = color_choices[i])
        #         ax.set_title(f'Mean and Std of {metric[0].capitalize() + metric[1:]}s of (True - Prediction)s for {var} Axis')
        #         ax.set_xlabel('Dead PMT Rate [%]')
        #         ax.set_ylabel(metric[0].capitalize() + metric[1:] + ' [cm]')
        #         ax.grid(True, linestyle='--', linewidth=0.5)
        #         fig.savefig(file_path + f'mean_{var}_axis_{metric}.png')

        #         plt.close(fig)
        


        


def analyze_multiple_regression(settings, sub_dir_names, percents):
    BASE_PATH = settings.mlPath
    df = pd.DataFrame(columns=['dead PMT rate', 'var', 'quantile', 'quantile error', 'median', 'median error'])


    for i, sub_dir in enumerate(sub_dir_names):
        ar = analysisResults(settings)
        print('BASE_PATH', BASE_PATH)
        print('particleLabel', settings.particleLabel)
        print('inputPath', settings.inputPath)
        print('fitqunPath', settings.fitqunPath)

        print('target', settings.target)
        settings.mlPath = BASE_PATH + sub_dir + '/'
        print('mlPath', settings.mlPath)
        single_ml_analysis, multi_ml_analysis = analyze_ml_regression(settings) 
        #print(f"SINGLE ANALYSIS: {single_ml_analysis}")
        #print(f"MULTI ANALYSIS: {multi_ml_analysis}")
        # ar.add_global_perf("ML", single_ml_analysis[0], single_ml_analysis[1], single_ml_analysis[2], single_ml_analysis[3], single_ml_analysis[4])
        # ar.add_var_perf("ML",multi_ml_analysis) # for now, only single analysis

        for j in range(len(single_ml_analysis[0])):
            new_row = {'dead PMT rate': percents[i], 'var': single_ml_analysis[0][j], 'quantile': single_ml_analysis[1][j], 'quantile error': single_ml_analysis[2][j], 'median': single_ml_analysis[3][j], 'median error': single_ml_analysis[4][j]}
            df = pd.concat([df, pd.DataFrame(new_row, index=[0])], ignore_index=True)
    
    df_summary = df.groupby(by=['dead PMT rate', 'var']).describe()
    print('summary stats', df_summary)

    df.to_csv(settings.outputPlotPath + 'reg_analysis_metrics.csv', sep=',', index=False)
    df_summary.to_csv(settings.outputPlotPath + 'reg_analysis_metrics_summary_stats.csv', sep=',', index=False)
    
    # save_boxplots(settings, df)






# def save_boxplots(settings, df=None, file_path=None):
#     if df is None and file_path is None:
#         return

#     if file_path is not None:
#         df = pd.read_csv(file_path)

#     color_choices=[
#     '#1f77b4', '#ff7f0e', '#2ca02c', 
#     '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
#     '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
#     '#17becf', '#9edae5']
    
#     for m in df.columns[2:]: # quantile, quantile error, median, ...
#         for c, v in enumerate(df['var'].unique()):
#             sns.boxplot(x = 'dead PMT rate', y = m, data=df[df['var'] == v], color=color_choices[c])
#             # sns.boxplot(x='dead PMT rate', y=m, data=df[df['var'] == v], orient='h')
#             filtered_df = df[df['var'] == v]
#             # sns.boxplot(x=filtered_df['dead PMT rate'], y=filtered_df[m], orient='h')
            
            
#             plt.title(f'{m[0].capitalize() + m[1:]} for {v} Axis by Dead PMT Rate')
#             plt.xlabel('Dead PMT Rate (%)')
#             plt.ylabel(m)

#             plt.savefig(f'{settings.outputPlotPath}positions_{v}_ML_{m[0].capitalize() + m[1:]}_boxplot.png')
#             plt.close()


# def flatten_dict(d):
#     """
#     Flatten a nested dictionary into a flat dictionary.
#     """
#     flat_dict = {}
#     for key, value in d.items():
#         if isinstance(value, dict):
#             # Recursively flatten nested dictionaries
#             flat_value = flatten_dict(value)
#             for subkey, subvalue in flat_value.items():
#                 flat_dict[str(key) + '.' + str(subkey)] = subvalue
#         else:
#             flat_dict[key] = value
#     return flat_dict

# def generate_summary(ar_list, percents):
#     percents_unique = list(set(percents))
#     summary = {}
#     for unique_p in percents_unique:
#         summary[unique_p] = {}
#         for a in ['X', 'Y', 'Z', 'Longitudinal', 'Transverse', 'Global']:
#             summary[unique_p][a] = {}
#             for v in ['median', 'median error', 'quantile', 'quantile error']:
#                 summary[unique_p][a][v] = compute_summary_stats(ar_list, percents, unique_p, a, v)
#     return summary



# def compute_summary_stats(ar_list, percents, target_p, axis: str, value: str):
#     filtered = [ar for ar, p in zip(ar_list, percents) if p == target_p]
    
    
#     values = []
#     for ar in filtered:
#         try:
#             values.append(ar.get_global_perf('ML', axis=axis, value=value))
#             # print('success')
#         except KeyError:
#             print('key error')
#             continue
#         except Exception as e:
#              print(f"An error occurred: {e}")

#     if not values:
#         print('returning none')
#         return None
   
#     values = np.array(values)
#     q25, median, q75 = np.percentile(values, [25, 50, 75])
    
#     summary_stats = {
#         'count': len(values),
#         'mean': np.mean(values),
#         'std_dev': np.std(values),
#         'min': np.min(values),
#         '25%': q25,
#         '50%': median,
#         '75%': q75,
#         'max': np.max(values),
#     }
    
#     return summary_stats