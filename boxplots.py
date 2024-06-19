import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# ROC_type = 'Muon_vs_Electron'
# ROC_type = 'Muon_vs_Pion'
ROC_type = 'Electron_vs_Muon'


# data_path = '/data/thoriba/t2k/plots/oct20_eMuPosPion_0dwallCut_flat_1_reg_2/'
data_path = '/data/thoriba/t2k/eval/oct20_eMuPosPion_0dwallCut_flat_1/09052024-171021_regress/22032024-142255/loss_log_20240608061238.csv'
# data_path = 
file_name = ROC_type + '_AUCs_all.csv'
# df = pd.read_csv(data_path+file_name, header=0)
df = pd.read_csv(data_path, header=None)
df.columns = ['rate', 'loss']

print(df)

# # sns.boxplot(x='# percent', y=' AUC', data=df)
# sns.boxplot(x='rate', y='loss', data=df)

plt.errorbar()


# ROC_type_title = ROC_type.replace('_', ' ')
# # plt.title(f'{ROC_type_title} AUC by Dead PMT Rate')
# # plt.xlabel('Dead PMT Rate (%)')
# # plt.ylabel('AUC')


plt.title(f'Regression Evaluation Loss by Dead PMT Rate')
plt.xlabel('Dead PMT Rate (%)')
plt.ylabel('Loss')

# plt.ylim(0, 1)

# Display the plot
plt.show()

plt.savefig(f'regression_loss_eb.png')


# summary stats

# grouped = df.groupby('rate')
# sum_stats_by_rate = grouped['loss'].describe()
# sum_stats_by_rate.to_csv('sum_stats_loss_rate.csv')

# print(grouped['loss'].describe())