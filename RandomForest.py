from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
import pandas as pd
import os
import re
import time
from argparse import ArgumentParser


PARAMETER_DICT = {
    'n_estimators': [100, 150, 200, 300, 400],
    'max_depth': [None, 4, 6, 8, 10],
    'min_samples_split': [2, 0.05, 0.1, 0.2],
    'min_samples_leaf': [1, 4, 6, 8, 10, 20],
    'max_features': ['auto', 'sqrt'],
    'max_samples': [None, 0.3, 0.4, 0.5]
}


def load_columns(file: str, *features, sep=','):
    """
    Read file into DataFrame using only the specified columns.

    :param file: Path to file.
    :param features: The columns that should be read.
    :param sep: The separator of the file.
    :return: Pandas DataFrame extracted from file
    """
    cols = features
    data = pd.read_csv(file, usecols=cols, sep=sep)
    return data


def extract_feature_vals(file: str, features: list, sep: str = ',', lang: str = 'L2' ):
    """
    Extracts feature values from a file with reading measures.

    :param file: The reading measures file from which the feature values should be extracted
    :param features: A list of feature names to be included
    :param sep: The separator used in the input file
    :returns: A dictionary with reading measures as key and the and an appropriate value (mostly average) in a text as
              values.
    """

    df = pd.read_csv(file, sep=sep)
    feature_cols = []
    if features:
        for feature in features:
            col = df[feature]
            col = col.replace(r'.', '0').astype(int)
            feature_cols.append(col)
    else:
        for column in range(3, df.shape[1]):
            col = df.iloc[:, column]
            col = col.replace(r'.', '0').astype(int)
            feature_cols.append(col)
    ordinal_features = ['LP', 'SL_in', 'SL_out', 'TRC_out', 'TRC_in']
    agg_features = {col.name: [sum(col) / len(col)] for col in feature_cols if col.name not in ordinal_features}
    for col in feature_cols:
        if col.name == 'SL_in' or col.name == 'SL_out':
            # gives average absolute saccade lengths
            agg_features[col.name] = sum(abs(col[col != 0])) / len(col[col != 0])
        if col.name == 'TRC_in' or col.name == 'TRC_out':
            agg_features[col.name] = sum(col) / len(col)
    agg_features['participant'] = re.search('ge_[a-z]{2}_\d\d', file)[0]
    agg_features['file'] = file
    agg_features['text'] = re.search('text_(\d\d?)', file)[0] + '_' + lang
    return agg_features


def construct_training_data(language: str, label_file: str = 'labels.csv', features: list = None,
                            text_features: bool = False):
    """
    Constructs the training data.

    :param data_directory: Path to directory with reading measure files.
    :param label_file: The file containing the labels.
    :param features: The reading measures to be used.
    :return: A Pandas data frame containing the aggregated features for each reading measure file.
    """
    feature_df = pd.DataFrame()
    if language == 'both':
        directory = 'data_L1'
        for file in os.listdir(directory):
            feature_vals = extract_feature_vals(f'{directory}/{file}', features, lang='L1')
            frame = pd.DataFrame(feature_vals)
            feature_df = pd.concat([feature_df, frame], ignore_index=True)
        directory = 'data_L2'
        for file in os.listdir(directory):
            feature_vals = extract_feature_vals(f'{directory}/{file}', features, lang='L2')
            frame = pd.DataFrame(feature_vals)
            feature_df = pd.concat([feature_df, frame], ignore_index=True)
    else:
        directory = 'data_' + language
        for file in os.listdir(directory):
            feature_vals = extract_feature_vals(f'{directory}/{file}', features, lang=language)
            frame = pd.DataFrame(feature_vals)
            feature_df = pd.concat([feature_df, frame], ignore_index=True)

    if text_features:
        tfs = pd.read_csv('utils/text_features.csv')
        feature_df = feature_df.merge(tfs)

    labels = load_columns(label_file, 'SUBJ_ID', 'MET Total')
    labels.set_index('SUBJ_ID', inplace=True)
    lookup = labels['MET Total'].to_dict()
    feature_df['MET_Score'] = feature_df['participant'].map(lookup)

    return feature_df


def get_args():
    parser = ArgumentParser()
    parser.add_argument('--language', '-la', type=str, default='L2', choices=['L1', 'L2', 'both'],
                        help='Path to data directory')
    parser.add_argument('--labels', '-l', type=str, default='labels.csv', help='Path to file containing labels')
    parser.add_argument('--measures', '-m', type=str, nargs='*', help='Specify the reading measures to be used.'
                                                                      'If none are given, all measures are used.')
    parser.add_argument('--split', '-s', type=str, default='random', choices=['random', 'text', 'participant'],
                        help='Specify the kind of split, using random, text or participant. Random splits the data set'
                             'randomly, text ensures that individual texts are not split between train and test sets,'
                             'and participant does the same for participants')
    parser.add_argument('--text_features', '-t', action='store_true', help='Whether text features should be used.')
    args = parser.parse_args()
    return args


def main():
    start_time = time.time()
    args = get_args()
    print(args)
    data = construct_training_data(args.language, args.labels, args.measures)
    data = data[data['MET_Score'] != 0]
    data.dropna(inplace=True)
    length = len(data['MET_Score'])
    print(f'Using {length} data points.')
    X = data.drop(columns=['file', 'participant', 'text', 'MET_Score'])
    y = data['MET_Score']

    random_forest = RandomForestRegressor(random_state=42)
    if args.split == 'random':
        cv = KFold(n_splits=10, shuffle=True)
    else:
        cv = GroupKFold(n_splits=10).split(X, y, data[args.split])

    print(f'Splitting on group {args.split}.')
    best_model = GridSearchCV(random_forest, param_grid=PARAMETER_DICT, verbose=1, cv=cv,
                              scoring='neg_mean_squared_error', n_jobs=-1)
    best_model.fit(X, y)
    print(f'Best parameters: {best_model.best_params_}')
    print(f'Best model avg cv score: {best_model.best_score_}')
    print(f'Time: {(time.time()-start_time)/60} mins')


if __name__ == '__main__':
    main()
