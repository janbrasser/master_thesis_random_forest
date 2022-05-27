import RandomForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, GroupKFold, KFold
from sklearn.dummy import DummyRegressor
from scipy.stats import ttest_ind
from datetime import date
import time
from statistics import mean, stdev
from matplotlib import pyplot as plt


def main():
    start_time = time.time()
    args = RandomForest.get_args()
    data = RandomForest.construct_training_data(args.language, args.labels, args.measures, args.text_features)
    data = data[data['MET_Score'] != 0]
    data.dropna(inplace=True)
    X = data.drop(columns=['file', 'participant', 'text', 'MET_Score'])
    y = data['MET_Score']
    nr_of_data_points = len(data['MET_Score'])
    print(f'Using {nr_of_data_points} data points.')

    random_forest = RandomForestRegressor(random_state=42)
    nr_of_texts = {'L1': 13, 'L2': 12, 'both': 12}
    if args.split == 'random':
        inner_cv = KFold(n_splits=4, shuffle=True)
        outer_cv = KFold(n_splits=10, shuffle=True)
    elif args.split == 'participant':
        inner_cv = GroupKFold(n_splits=4)
        outer_cv = GroupKFold(n_splits=10)
    else:
        inner_cv = GroupKFold(n_splits=4)
        outer_cv = GroupKFold(n_splits=nr_of_texts[args.language])

    cvd = GridSearchCV(random_forest, param_grid=RandomForest.PARAMETER_DICT, verbose=1, cv=inner_cv,
                       scoring='neg_mean_squared_error', n_jobs=5)

    if args.split == 'random':
        model_score = cross_val_score(cvd, X, y, cv=outer_cv, scoring='neg_mean_squared_error', n_jobs=5)
    else:
        model_score = cross_val_score(cvd, X, y, cv=outer_cv, groups=data[args.split],
                                      scoring='neg_mean_squared_error', fit_params={"groups": data[args.split]},
                                      n_jobs=5)

    baseline = DummyRegressor()
    if args.split == 'random':
        baseline_score = cross_val_score(baseline, X, y, cv=outer_cv, scoring='neg_mean_squared_error', n_jobs=5)
    else:
        baseline_score = cross_val_score(baseline, X, y, cv=outer_cv, groups=data[args.split],
                                         scoring='neg_mean_squared_error', n_jobs=-1)

    forest = RandomForestRegressor()
    if args.split == 'random':
        cv = KFold(n_splits=10, shuffle=True)
    else:
        cv = GroupKFold(n_splits=10).split(X, y, data[args.split])

    print(f'Splitting on group {args.split}.')
    best_model = GridSearchCV(forest, param_grid=RandomForest.PARAMETER_DICT, verbose=1, cv=cv,
                              scoring='neg_mean_squared_error', n_jobs=-1)
    best_model.fit(X, y)

    today = date.today()
    file_date = today.strftime("%Y_%m_%d")

    predictions = best_model.predict(X)
    fig, ax = plt.subplots()
    ax.plot(predictions, y, '.b', zorder=10)
    lims = [min([ax.get_xlim(), ax.get_ylim()]), max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, '-r', zorder=0)
    ax.set_aspect('equal')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('True Label')
    fig.savefig(f'outputs/{file_date}_{args.language}_{args.split}_{args.text_features}.jpg')

    today = date.today()
    file_date = today.strftime("%Y_%m_%d")
    with open(f'outputs/{file_date}_{args.language}_{args.split}_{args.text_features}.txt', 'w', encoding='utf-8') as f:
        f.write(f'Date: {file_date}\n\n')
        f.write(f'Arguments: {args}\n\n')
        f.write(f'Number of data points: {nr_of_data_points}\n\n')
        f.write(f'Model scores: {model_score}\n Mean: {mean(model_score)}\tSTD: {stdev(model_score)}\t'
                f'SE: {stdev(model_score)/(len(model_score)**1/2)}\n\n')
        f.write(f'Baseline scores: {baseline_score}\n Mean: {mean(baseline_score)}\tSTD: {stdev(baseline_score)}\t'
                f'SE: {stdev(baseline_score)/(len(baseline_score)**1/2)}\n\n')
        f.write(f't-Test: {ttest_ind(model_score, baseline_score)}\n\n')
        f.write(f'Best model parameters on full refit: {best_model.best_params_}\n\n')
        f.write('Model prediction\ttrue label\n')
        for prediction, label in zip(predictions, y):
            f.write(f'{prediction}\t{label}\n')
    print(f'Time: {round((time.time() - start_time) / 60, 2)} mins')



if __name__ == '__main__':
    main()
