# implement later
from scipy import stats
from analyze_output.analyze_ml_regression import analyze_ml_regression, save_residual_plot, save_residual_residual_plot #, analyze_energy_long
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
        self.BASE_PATH = settings.mlPath

        self.bias_raw_df = None
        self.bias_summary_df = None

        # initialize analysis dictionary
        for p in list(set(percents)):
            self.analysis_dict[p] = []

        print('constructor done')
    
    def analyze(self, tasks=['all']):
        self.compute_bias_summary_stats()
        self.plot_errorbars()

    def plot_resdiual_scatter(self, feature_name='energy', v_axis='Longitudinal'):
        """
        Plots a scatter plot of residuals versus a specified feature, and saves it in outputPlotPath specified in analysis_config.ini, for all sub_directories provided to constructor

        This method generates a scatter plot where the x-axis represents the values of the specified feature from the dataset,
        and the y-axis represents the residuals (the differences between the actual values and the predicted values) with respect to axis specified in `v_axis`. 

        Parameters:
        -----------
        feature_name : str
            The name of the feature (column) in the dataset for which the residual scatter plot will be generated.
            It must be one of ['energy', 'visible energy', 'towall', 'total_charge', 'nhit']
        v_axis : str
            The name of the column that represents the residuals in the dataset. 

        Returns:
        --------
        None
            This method does not return any value. It directly generates and displays the scatter plot.
        """
        settings = self.settings
        percents = self.percents
        sub_dir_names = self.sub_dirs

        # BASE_PATH = settings.mlPath
        
        # df = pd.DataFrame(columns=['dead PMT rate', 'var', 'quantile', 'quantile error', 'median', 'median error'])

        for i, sub_dir in enumerate(sub_dir_names):
            print('BASE_PATH', self.BASE_PATH)
            print('particleLabel', settings.particleLabel)
            print('inputPath', settings.inputPath)
            print('fitqunPath', settings.fitqunPath)

            print('target', settings.target)
            settings.mlPath = self.BASE_PATH + sub_dir + '/'
            print('mlPath', settings.mlPath)
            # single_ml_analysis, multi_ml_analysis = analyze_ml_regression(settings) 
            # analyze_energy_long(settings, feature_name, v_axis)
            save_residual_plot(settings, feature_name, v_axis)
    
    def plot_residual_residual_scatter(self, targets=['positions', 'momenta'], axes=['Longitudinal', 'Global'], ml_paths=None):
        if ml_paths is None:
            print('ml_paths should be a list of 2 paths to directories where evaluation outputs (indices.npy, pred_xxx.npy, etc) are stored')
            print("Example: ml_paths = ['/data/username/model/reg1/', '/data/username/model/reg1/']")
            return
        settings = self.settings
        percents = self.percents
        sub_dir_names = self.sub_dirs
        BASE_PATH = settings.mlPath
        # for i, sub_dir in enumerate(sub_dir_names):
        # print('BASE_PATH', BASE_PATH)
        print('-----------plot_residual_residual_scatter_plot------------------')
        print('particleLabel', settings.particleLabel)
        print('inputPath', settings.inputPath)
        print('fitqunPath', settings.fitqunPath)
        print('targets', targets)
        print('axes for targets', axes)
        # settings.mlPath = BASE_PATH + sub_dir + '/'
        # print('mlPath', settings.mlPath)
        print('from evaluation outputs stored in ', ml_paths)
        # single_ml_analysis, multi_ml_analysis = analyze_ml_regression(settings) 
        # analyze_energy_long(settings, feature_name, v_axis)
        save_residual_residual_plot(settings, targets, axes, ml_paths)

        

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

    def plot_errorbars_multi_models(self, file_paths=None, unit=None, labels=None):
        if file_paths is None:
            self.plot_errorbars()
            return

        color_choices=[
            '#1f77b4', '#ff7f0e', '#2ca02c', 
            '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
            '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d',
            '#17becf', '#9edae5']
        
        plt.rcdefaults()
        
        if unit is None:
            if self.settings.target in ['momenta', 'momentum', 'mom']:
                unit = '%'
            else:
                unit = 'cm'
        
        if unit == '%':
            percentify = 100
        else:
            percentify = 1
        
        dfs = []
        for file_path in file_paths:
            dfs.append(pd.read_csv(file_path))
        
        # this means axis name. can be more defensive
        axis_names = dfs[0]['var'].unique()

        for i, var in enumerate(axis_names):
            
            for metric in ['quantile', 'median']:
                fig, ax = plt.subplots()

                for j, df in enumerate(dfs):
                    
                    dfsem = df[df['var'] == var].groupby(by=['dead PMT rate', 'var']).sem(ddof=0).reset_index()
                    dfmean = df[df['var'] == var].groupby(by=['dead PMT rate', 'var']).mean().reset_index()
                    
                    # Replace NaN values 
                    for r in range(len(dfsem)):
                        if dfsem.iloc[r].isnull().values.any():
                            # if it does, print the row
                            # print(dfsem.iloc[r])
                            # substitude the row with values from original df. For quantile, we use quantile error from original df
                            dfsem.loc[r, 'quantile'] = df[df['var'] == var]['quantile error'].loc[r]
                            dfsem.loc[r, 'median'] = df[df['var'] == var]['median error'].loc[r]


                    x = dfmean['dead PMT rate']
                    y = dfmean[metric]*percentify

                    # # apply filter (aka cut / mask)
                    df_reg = df[df['dead PMT rate'] <= 3]
                    x_reg = df_reg['dead PMT rate']
                    y_reg = df_reg[metric] * percentify

                    slope, intercept, r_value, _, _ = stats.linregress(x_reg, y_reg)
                    # Create a line of best fit
                    line = slope * x + intercept


                    ax.errorbar(x, y, yerr=dfsem[metric]*percentify,
                                 fmt='o', color = color_choices[j], capsize=5, label = labels[j] + f' (Slope: {slope:.3f})')
                    ax.plot(x, line, color=color_choices[j], linestyle='--')


                if self.settings.target in ['momentum', 'momenta', 'mom']:
                    title_str = f'{metric[0].capitalize() + metric[1:]} of (True - Pred) / True Momenta for {var} Axis'
                else:
                    title_str = f'{metric[0].capitalize() + metric[1:]} of (True - Pred) {self.settings.target[0].capitalize() + self.settings.target[1:]} for {var} Axis'
    
                ax.set_title(title_str)
                ax.set_xlabel('Dead PMT Rate [%]')
                
                if 'angle' in var.lower():
                    unit = 'deg'

                if metric == 'median':
                    ax.set_ylabel(var + ' Bias ' + f' [{unit}]')
                elif metric == 'quantile':
                    ax.set_ylabel(var + ' Resolution ' + f' [{unit}]')

                ax.set_xlim([None, 4])
                ax.set_ylim([None, 4.2])
                
                ax.grid(True, linestyle='--', linewidth=0.5)
                ax.legend()
                fig.savefig(self.settings.outputPlotPath + f'{self.settings.target}_{var}_axis_{metric}_errorbars_.png')
                # fig.clear()

        
        


    def plot_errorbars(self, file_path=None, unit=None):
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
        
        plt.rcdefaults()
        
        if unit is None:
            if self.settings.target in ['momenta', 'momentum', 'mom']:
                unit = '%'
            else:
                unit = 'cm'
        
        if unit == '%':
            percentify = 100
        else:
            percentify = 1
        

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

                plt.errorbar(dfmean['dead PMT rate'], y=dfmean[metric]*percentify, yerr=dfsem[metric]*percentify, fmt='o', color = color_choices[i], capsize=5)

                if self.settings.target in ['momentum', 'momenta', 'mom']:
                    title_str = f'{metric[0].capitalize() + metric[1:]} of (True - Pred) / True Momenta for {var} Axis'
                else:
                    title_str = f'{metric[0].capitalize() + metric[1:]} of (True - Pred) {self.settings.target[0].capitalize() + self.settings.target[1:]} for {var} Axis'

                ax.set_title(title_str)
                ax.set_xlabel('Dead PMT Rate [%]')
                
                if 'angle' in var.lower():
                    unit = 'deg'

                ax.set_ylabel(metric[0].capitalize() + metric[1:] + f' [{unit}]')
                ax.grid(True, linestyle='--', linewidth=0.5)
                fig.savefig(self.settings.outputPlotPath + f'{self.settings.target}_mean_{var}_axis_{metric}.png')

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


def compare_residuals_scatter():
    pass




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