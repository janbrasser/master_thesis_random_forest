import pandas as pd

df_all = pd.concat([pd.read_csv('lexical_features_fixed_de.csv'), pd.read_csv('lexical_features_fixed_en.csv')], axis=0)
print(df_all)

df_all = df_all[['text', 'sentence', 'surpGPT']]
df_all = df_all.groupby(['text', 'sentence'], as_index=False).mean().groupby(['text'], as_index=False).mean()
df_all[['text', 'surpGPT']].to_csv('surprisals.csv')
