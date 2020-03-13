import datetime
import re

import pandas as pd
import numpy as np
import tqdm
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor  # инструмент для создания и обучения модели
from sklearn import metrics  # инструменты для оценки точности модели


# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 42


VERSION = 11
DIR_TRAIN = 'files/'  # подключил к ноутбуку свой внешний датасет
DIR_TEST = 'files/'
VAL_SIZE = 0.33   # 33%
N_FOLDS = 5

# CATBOOST
ITERATIONS = 2000
LR = 0.1


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''

    df_output = df_input.copy()

    # ################### Предобработка ##############################################################
    # убираем не нужные для модели признаки
    df_output.drop(['Таможня', 'Состояние', 'id'], axis=1, inplace=True, )

    # ################### fix ##############################################################
    # Переводим признаки из float в int (иначе catboost выдает ошибку)
    for feature in ['modelDate', 'numberOfDoors', 'mileage', 'productionDate']:
        df_output[feature] = df_output[feature].astype('int32')

    # ################### Feature Engineering ####################################################
    # тут ваш код на генерацию новых фитчей
    # ....

    # ################### Clean ####################################################
    # убираем признаки которые еще не успели обработать,
    df_output.drop(['Комплектация', 'description', 'Владение'], axis=1, inplace=True, )

    return df_output


BODY_TYPES = {
    'седан': ['седан'],
    'внедорожник': ['внедорожник'],
    'хэтчбек': ['хэтчбек'],
    'лифтбек': ['хэтчбек', 'лифтбек'],
    'универсал': ['универсал'],
    'минивэн': ['минивэн'],
    'компактвэн': ['минивэн', 'компактвэн'],
    'купе': ['купе', 'седан'],
    'пикап': ['пикап', 'внедорожник'],
    'фургон': ['фургон'],
    'кабриолет': ['кабриолет', 'седан'],
    'родстер': ['родстер', 'купе', 'кабриолет', 'седан'],
    'микровэн': ['микровэн'],
    'лимузин': ['лимузин', 'седан']
}


def set_body(x):
    for t_name, t_arr in BODY_TYPES.items():
        if t_name in x:
            return t_arr
    return x


def set_name(x):
    if 'Electro' in x:
        return ''
    return re.sub(r'[\d]{1}[.]{1}\d{1}[a-z]*[\s]{1}[A-Z]+[\s]{1}[a-zа-я.()\s\d]+[A-Z]*$', '', x).strip()


def main(all_=True, new=True, train=True):
    df_train = pd.read_csv(DIR_TRAIN + 'test.csv')  # мой подготовленный датасет для обучения модели
    # df_test = pd.read_csv(DIR_TEST + 'test.csv')
    sample_submission = pd.read_csv(DIR_TEST + 'sample_submission.csv')

    train_preproc = preproc_data(df_train)
    # X_sub = preproc_data(sample_submission)

    # train_preproc.drop(['Привод', 'Руль', 'Владельцы', 'ПТС', 'enginePower'], axis=1, inplace=True, )  # убрал лишний столбец, которого нет в testе

    X = train_preproc
    y = sample_submission.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True,
                                                        random_state=RANDOM_SEED)

    pd.options.mode.chained_assignment = None
    X_train['bodyType'] = X_train['bodyType'].apply(set_body)
    # X_train['transmittion'] = X_train['vehicleConfiguration'].apply(lambda x: x.split()[1])
    X_train['motor'] = X_train['engineDisplacement'].apply(lambda x: float(x.replace(' LTR', '')) if x.replace(' LTR', '').replace('.', '').isdigit() else np.nan)
    X_train['enginePower'] = X_train['enginePower'].apply(lambda x: int(x.replace(' N12', '')))

    # X_train.drop(['vehicleConfiguration'], axis=1, inplace=True)

    for index, row in X_train.iterrows():
        for bt in row['bodyType']:
            X_train.at[index, bt] = 1

    for b_name in BODY_TYPES:
        X_train[b_name].fillna(0, inplace=True)

    X_train['model'] = X_train['name'].apply(set_name)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(X_train.head())

    X_train = pd.get_dummies(X_train, columns=['Привод', 'Руль', 'Владельцы', 'ПТС', 'vehicleTransmission', 'fuelType', 'color', 'brand'])

    # чтобы не писать весь список этих признаков, просто вывел их через nunique(). и так сойдет)
    # X_train.nunique()

    # Keep list of all categorical features in dataset to specify this for CatBoost
    # cat_features_ids = np.where(X_train.apply(pd.Series.nunique) < 3000)[0].tolist()

    # model = CatBoostRegressor(iterations=ITERATIONS,
    #                           learning_rate=LR,
    #                           random_seed=RANDOM_SEED,
    #                           eval_metric='MAPE',
    #                           custom_metric=['R2', 'MAE']
    #                           )
    #
    # model.fit(X_train, y_train,
    #           cat_features=cat_features_ids,
    #           eval_set=(X_test, y_test),
    #           verbose_eval=100,
    #           use_best_model=True,
    #           plot=True
    #           )

    # model.save_model('catboost_single_model_baseline.model')
    # test_predict = model.predict(X_test)
    # test_score = mape(y_test, test_predict)
    #
    # print(mape(y_test, test_predict), '---------------')
    # predict_submission = model.predict(X_sub)
    # predict_submission

    # sample_submission['price'] = predict_submission
    # sample_submission.to_csv(f'submission_v{VERSION}.csv', index=False)
    # sample_submission.head(10)

    # submissions = pd.DataFrame(0, columns=["sub_1"],
    #                            index=sample_submission.index)  # куда пишем предикты по каждой модели
    # score_ls = []
    # splits = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED).split(X, y))
    #
    # def cat_model(y_train, X_train, X_test, y_test):
    #     model = CatBoostRegressor(iterations=ITERATIONS,
    #                               learning_rate=LR,
    #                               eval_metric='MAPE',
    #                               random_seed=RANDOM_SEED, )
    #     model.fit(X_train, y_train,
    #               cat_features=cat_features_ids,
    #               eval_set=(X_test, y_test),
    #               verbose=False,
    #               use_best_model=True,
    #               plot=False)
    #
    #     return (model)
    #
    # def mape(y_true, y_pred):
    #     return np.mean(np.abs((y_pred - y_true) / y_true))
    #
    # for idx, (train_idx, test_idx) in tqdm(enumerate(splits), total=N_FOLDS, ):
    #     # use the indexes to extract the folds in the train and validation data
    #     X_train, y_train, X_test, y_test = X.iloc[train_idx], y[train_idx], X.iloc[test_idx], y[test_idx]
    #     # model for this fold
    #     model = cat_model(y_train, X_train, X_test, y_test, )
    #     # score model on test
    #     test_predict = model.predict(X_test)
    #     test_score = mape(y_test, test_predict)
    #     score_ls.append(test_score)
    #     print(f"{idx + 1} Fold Test MAPE: {mape(y_test, test_predict):0.3f}")
    #     # submissions
    #     submissions[f'sub_{idx + 1}'] = model.predict(X_sub)
    #     model.save_model(f'catboost_fold_{idx + 1}.model')
    #
    # print(f'Mean Score: {np.mean(score_ls):0.3f}')
    # print(f'Std Score: {np.std(score_ls):0.4f}')
    # print(f'Max Score: {np.max(score_ls):0.3f}')
    # print(f'Min Score: {np.min(score_ls):0.3f}')


if __name__ == '__main__':
    main(train=False, new=False)
