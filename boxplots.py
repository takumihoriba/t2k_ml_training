import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

# data_path = '/home/thoriba/t2k2/t2k_ml_training/all_data.csv'

auc_error_bar = 0

# acc_loss_classification_error_bar = 1

if 1:
    # for acc and loss
    m = 'accuracy' 
    # m = 'loss'
    data_path = f'/data/thoriba/t2k/models/14042024-000620/{m}_summary_stats_per_percent20240606022134.csv'
    df = pd.read_csv(data_path, header=None)
    df.columns = ['rate', 'min', 'q1', 'med', 'mean', 'q3', 'max', 'sd']

    plt.errorbar(x=df['rate'], y=df['mean'], yerr=df['sd'], fmt='o', color='b')

    plt.xlabel('Dead PMT Rate [%]')
    plt.ylabel(m)
    # auc_title = roc_code.replace('_', ' ')
    plt.title(f'{m} by dead PMT rate')

    plt.savefig(f'/data/thoriba/t2k/models/14042024-000620/classify_{m}_summary_error_bar_plot.png')



    print(df)

if auc_error_bar:
    roc_code = 'Electron_vs_Muon'
    data_path = f'/data/thoriba/t2k/plots/oct20_eMuPosPion_0dwallCut_flat_1_reg_2/{roc_code}_AUCs_all.csv'
    df = pd.read_csv(data_path)
    # df['dead PMT Rate'] = df['dead PMT Rate'].astype('str')

    df.columns = ['rate', 'auc']

    df_mean = df.groupby('rate').mean().reset_index()
    df_std = df.groupby('rate').std().reset_index()

    df_mean['std'] = df_std['auc']

    print(df_mean)



    plt.errorbar(x = df_mean['rate'], y = df_mean['auc'], yerr = df_mean['std'], fmt='o')
    plt.xlabel('Dead PMT Rate [%]')
    plt.ylabel('AUC')
    auc_title = roc_code.replace('_', ' ')
    plt.title(f'Mean and Standard Deviation of AUC ({auc_title})')

    plt.savefig(f'/data/thoriba/t2k/plots/oct20_eMuPosPion_0dwallCut_flat_1_reg_2/{roc_code}_auc_errorbar.png')






# df['dead PMT rate'] = pd.Categorical(df['dead PMT rate'])


# output_path = '/data/thoriba/t2k/plots/oct20_eMuPosPion_0dwallCut_flat_1_reg_multi/'


# sns.boxplot(x='dead PMT rate', y='quantile', data=df[df['var'] == 'X'])

# plt.savefig(f'regress_var_box_test.png')

# print(df['var'].unique())

# f, ax = plt.subplots(figsize=(7, 6))

# print()
# for m in df.columns[2:]: # quantile, quantile error, median, ...
#     for v in df['var'].unique():
#         sns.boxplot(x = 'dead PMT rate', y = m, data=df[df['var'] == v])
#         # sns.boxplot(x='dead PMT rate', y=m, data=df[df['var'] == v], orient='h')
#         filtered_df = df[df['var'] == v]
#         # sns.boxplot(x=filtered_df['dead PMT rate'], y=filtered_df[m], orient='h')
        
        
#         plt.title(f'{m[0].capitalize() + m[1:]} for {v} Axis by Dead PMT Rate')
#         plt.xlabel('Dead PMT Rate (%)')
#         plt.ylabel(m)

#         plt.savefig(f'{output_path}positions_{v}_ML_{m[0].capitalize() + m[1:]}_boxplot.png')
#         plt.close()
