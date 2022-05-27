import pandas as pd
import os

scores = []
names = []

for file in os.listdir('../met_scores'):
    df = pd.read_csv(f'../met_scores/{file}')
    scores.append(sum(df['accuracy'])/len(df['accuracy'])*100)
    names.append(file[:-4])

potsdam_frame = pd.DataFrame()
potsdam_frame['SUBJ_ID'] = names
potsdam_frame['MET Total'] = scores
zurich_frame = pd.read_csv('../labels.csv')
all_labels = pd.concat([zurich_frame, potsdam_frame])
all_labels.to_csv('../all_labels.csv')


