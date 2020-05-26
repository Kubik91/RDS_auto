import asyncio
import datetime
import json
import os
import re
import sys
import traceback
import urllib

import aiohttp
import pandas as pd
import numpy as np
import tqdm
from aiofile import AIOFile, Writer
from bs4 import BeautifulSoup
from catboost import CatBoostRegressor
from dateutil import relativedelta
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

URL = 'http://auto.drom.ru/archive/'

TRAIN_FILES = []

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
    'лимузин': ['лимузин', 'седан'],
    'тарга': ['тарга', 'купе', 'кабриолет', 'родстер']
}
WHEEL_DRIVE = {
    'передний': 1,
    'задний': 2,
    'полный': 3
}
RORL = {
    'Левый': 2,
    'Правый': 1
}
USERS = {
    '1\xa0владелец': 1,
    '2\xa0владельца': 2,
    '3 или более': 3
}
PTS = {
    'Дубликат': 1,
    'Оригинал': 2
}
TRANS = {
    'роботизированная': 4,
    'вариатор': 3,
    'автоматическая': 2,
    'механическая': 1
}
FUEL = {
    'бензин': 1,
    'дизель': 2,
    'электро': 4,
    'гибрид': 5,
    'газ': 6
}
COLOR = {
    'синий': 1,
    'чёрный': 2,
    'бежевый': 3,
    'белый': 4,
    'коричневый': 5,
    'серебристый': 6,
    'пурпурный': 7,
    'серый': 8,
    'красный': 9,
    'золотистый': 10,
    'зелёный': 11,
    'фиолетовый': 12,
    'голубой': 13,
    'оранжевый': 14,
    'жёлтый': 15,
    'розовый': 16
}
BRAND = {
    'AUDI': 1,
    'INFINITI': 2,
    'VOLKSWAGEN': 3,
    'VOLVO': 4,
    'SKODA': 5,
    'MERCEDES': 6,
    'NISSAN': 7,
    'MITSUBISHI': 8,
    'HONDA': 9,
    'TOYOTA': 10,
    'BMW': 11,
    'SUZUKI': 12,
    'LEXUS': 13
}

FRAMETYPES = {
    'седан': 10,
    'лифтбек': 9,
    'универсал': 3,
    'минивэн': 6,
    'пикап': 12,
    'купе': 1,
    'кабриолет': 11
}


def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true))


def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''

    pd.options.mode.chained_assignment = None

    df_output = df_input.copy()

    # ################### Предобработка ##############################################################
    # убираем не нужные для модели признаки
    # df_output.drop(['Таможня', 'Состояние', 'id'], axis=1, inplace=True, )

    # ################### fix ##############################################################
    # Переводим признаки из float в int (иначе catboost выдает ошибку)
    for feature in ['modelDate', 'numberOfDoors', 'mileage', 'productionDate']:
        df_output[feature] = df_output[feature].astype(np.int)

    # ################### Feature Engineering ####################################################
    # тут ваш код на генерацию новых фитчей
    # ....

    # ################### Clean ####################################################
    # убираем признаки которые еще не успели обработать,

    df_output['bodyType'] = df_output['bodyType'].apply(set_body)
    df_output['motor'] = df_output['engineDisplacement'].apply(
        lambda x: float(x.replace(' LTR', '')) if x.replace(' LTR', '').replace('.', '').isdigit() else np.nan
    ).fillna(0)
    df_output['enginePower'] = df_output['enginePower'].apply(lambda x: int(x.replace(' N12', ''))).astype(np.int)

    # df_output['Комплектация'].apply(set_complectation)

    for index, row in df_output.iterrows():
        for bt in row['bodyType']:
            df_output.at[index, bt] = 1
        for comp in json.loads(row['Комплектация'].replace("['", '', 1)[::-1].replace("']"[::-1], '', 1)[::-1]):
            df_output.at[index, comp['name']] = len(comp['values'])
            if comp['name'] not in COMP_ATTRS:
                COMP_ATTRS.append(comp['name'])

    for b_name in BODY_TYPES:
        df_output[b_name] = df_output[b_name].fillna(0).astype(np.int)

    for c_name in set(COMP_ATTRS):
        df_output[c_name] = df_output[c_name].fillna(0).astype(np.int)

    df_output['model'] = df_output['name'].apply(set_name)

    df_output['Привод'] = df_output['Привод'].apply(lambda x: WHEEL_DRIVE.get(x, 0)).astype(np.int)
    df_output['Руль'] = df_output['Руль'].apply(lambda x: RORL.get(x, 0)).astype(np.int)
    df_output['Владельцы'] = df_output['Владельцы'].apply(lambda x: USERS.get(x, 0)).astype(np.int)
    df_output['ПТС'] = df_output['ПТС'].apply(lambda x: PTS.get(x, 0)).astype(np.int)
    df_output['vehicleTransmission'] = df_output['vehicleTransmission'].apply(lambda x: TRANS.get(x, 0)).astype(np.int)
    df_output['fuelType'] = df_output['fuelType'].apply(lambda x: FUEL.get(x, 0)).astype(np.int)
    df_output['color'] = df_output['color'].apply(lambda x: COLOR.get(x, 0)).astype(np.int)
    df_output['brand'] = df_output['brand'].apply(lambda x: BRAND.get(x, 0)).astype(np.int)
    df_output['Таможня'] = df_output['Таможня'].apply(lambda x: 1 if x == 'Растаможен' else 0).astype(np.int)
    df_output['Состояние'] = df_output['Состояние'].apply(lambda x: 2 if x == 'Не требует ремонта' else 1).astype(np.int)
    df_output['Владение'] = df_output['Владение'].apply(get_ownership).astype(np.int)

    # print(df_output['Состояние'].unique())

    # for feature in ['brand', 'color', 'fuelType', 'modelDate', 'numberOfDoors', 'productionDate',
    #                 'vehicleTransmission', 'Привод', 'Руль', 'Состояние', 'Владельцы', 'ПТС', 'Таможня']:
    #     df_output[feature] = df_output[feature].astype('int32')

    df_output.drop(['vehicleConfiguration', 'description', 'bodyType', 'name', 'Комплектация', 'engineDisplacement'],
                   axis=1, inplace=True)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_output.columns)

    return df_output


def set_body(x):
    for t_name, t_arr in BODY_TYPES.items():
        if t_name in x:
            return t_arr
    return x


COMP_ATTRS = []


def set_complectation(x):
    globals()
    for comp in json.loads(x.replace("['", '', 1)[::-1].replace("']"[::-1], '', 1)[::-1]):
        if comp['name'] not in COMP_ATTRS:
            COMP_ATTRS.append(comp['name'])
    return x


def get_ownership(x):
    if type(x) == str:
        matches = re.match(r'(?P<year>\d+(?=(\s*лет)|(\s*год)))?\D*(?P<month>\d+(?=\s*мес))?', x)
        now = datetime.datetime.now()
        return ((now + relativedelta.relativedelta(years=int(matches.group('year') or 0), months=int(matches.group('month') or 0))) - now).days * 24 * 3600
    return 0


def set_name(x):
    if 'Electro' in x:
        return ''
    return re.sub(r'[\d]{1}[.]{1}\d{1}[a-z]*[\s]{1}[A-Z]+[\s]{1}[a-zа-я.()\s\d]+[A-Z]*$', '', x).strip()


sem = asyncio.Semaphore(20)
COUNTS = 0
LEFT = -1


async def get_page(url, lvl=0):
    try:
        async with sem:
            async with aiohttp.ClientSession() as session:
                response = await session.get(url)
                if response.status == 200:
                    return await response.text()
    except aiohttp.client_exceptions.ClientPayloadError:
        return await get_page(url, lvl+1) if lvl < 10 else None


def get_url(x, lvl=0):
    current_url = f"{URL}{(list(BRAND.keys())[list(BRAND.values()).index(x['brand'])]).lower().replace('mercedes', 'mercedes-benz')}/all/"

    query_params = {}

    if x['modelDate'] and lvl < 12:
        query_params.update({'minyear': x['modelDate']-max(lvl-5, 0), 'maxyear': x['modelDate']+max(lvl-5, 0)})
    if lvl < 11:
        for f_name, f_keys in FRAMETYPES.items():
            if x[f_name]:
                query_params.update({'frametype[]': f_keys})
                break
        else:
            if x['внедорожник']:
                if x['numberOfDoors'] in (3, 4):
                    query_params.update({'frametype[]': 8})
                else:
                    query_params.update({'frametype[]': 7})
            elif x['хэтчбек']:
                if x['numberOfDoors'] in (3, 4):
                    query_params.update({'frametype[]': 4})
                else:
                    query_params.update({'frametype[]': 5})

    if x['fuelType'] and lvl < 2:
        query_params.update({'fueltype': x['fuelType']})
    if x['Руль'] and lvl < 2:
        query_params.update({'w': x['Руль']})
    if x['ПТС'] and lvl < 3:
        query_params.update({'pts': x['ПТС']})
    if x['Привод'] and lvl < 6:
        query_params.update({'privod': x['Привод']})
    if x['Состояние'] and lvl < 9:
        query_params.update({'damaged': x['Состояние']})
    if x['enginePower'] and lvl < 4:
        query_params.update({'minpower': x['enginePower']-10*lvl, 'maxpower': x['enginePower']+10*lvl})
    if x['mileage'] and lvl < 7:
        query_params.update({'minprobeg': x['mileage']*1.609344-5000*(lvl+1), 'maxprobeg': x['mileage']*1.609344+5000*(lvl+1)})
    if x['motor'] and lvl < 8:
        query_params.update({'mv': x['motor']-.1*lvl, 'xv': x['motor']+.1*lvl})
    if x['vehicleTransmission'] and lvl < 5:
        query_params.update({'transmission': 1 if x['vehicleTransmission'] == 1 else 2})
    current_url = f'{current_url}?{urllib.parse.urlencode(query_params)}'
    return current_url


async def get_page_items(response):
    all_items = []
    soup = BeautifulSoup(response, "html.parser")
    all_items += [item.get('href') for item in soup.findAll('a', class_='b-advItem b-advItem_removed')]
    next_page = soup.find('a', class_='b-pagination__item b-link b-link_type_complex b-pagination__item_next')
    while next_page:
        response = await get_page(next_page.get('href'))
        if response:
            soup = BeautifulSoup(response, "html.parser")
            next_page = soup.find('a', class_='b-pagination__item b-link b-link_type_complex b-pagination__item_next')
            all_items += [item.get('href') for item in soup.findAll('a', class_='b-advItem b-advItem_removed')]
        else:
            break
    return all_items


async def get_item(x, all):
    # try:
    if f'data_{x["id"]}.csv' not in TRAIN_FILES:
        current_url = get_url(x)
        response = await get_page(current_url)
        all_items = []
        lvl = 0
        data = None
        if response:
            all_items = await get_page_items(response)
            if len(all_items) < 15:
                while len(all_items) < 15 and lvl <= 12:
                    lvl += 1
                    current_url = get_url(x, lvl)
                    response = await get_page(current_url)
                    if response:
                        all_items = await get_page_items(response)
                    else:
                        all_items = []
                if len(all_items) > 50:
                    lvl -= 1
                    current_url = get_url(x, lvl)
                    response = await get_page(current_url)
                    if response:
                        all_items = await get_page_items(response)
                    else:
                        all_items = []
                    if len(all_items) < 10:
                        lvl += 1
                        current_url = get_url(x, lvl)
                        response = await get_page(current_url)
                        if response:
                            all_items = await get_page_items(response)

        for item in all_items:
            try:
                new_data = await get_for_test_item(item, x, lvl)
                if data is None:
                    data = pd.DataFrame(columns=new_data.keys())
                if new_data is None:
                    raise Exception('new_data is None')
                data = data.append(new_data, ignore_index=True)
            except Exception as e:
                data = None
                print()
                print(f'{e=}')
                print(traceback.print_tb(e.__traceback__))
                # return
                break

        if data is not None:
            data_str = data.to_csv(encoding='utf-8', index=False)
            async with AIOFile(f'train/data_{x["id"]}.csv', 'w') as afp:
                writer = Writer(afp)
                await writer(data_str)
                await afp.fsync()

    global COUNTS, LEFT
    COUNTS += 1
    complete = int((COUNTS*100)/all)
    if not COUNTS == complete:
        LEFT = complete
    sys.stdout.write(f'Ход выполнения: {LEFT}%({COUNTS} из {all})\r')
    sys.stdout.flush()


async def get_for_test(x, all):
    global COUNTS, LEFT
    COUNTS += 1
    complete = int((COUNTS*100)/all)
    if not COUNTS == complete:
        LEFT = complete
    sys.stdout.write(f'Ход выполнения: {LEFT}%({COUNTS} из {all})\r')
    sys.stdout.flush()


async def get_for_tests():
    print('--start--')
    responses = await asyncio.gather(*[get_for_test(i, len(list(range(1, 24)))) for i in range(1, 24)])
    print()
    print('--finish--')


async def get_for_items(X):
    global TRAIN_FILES
    TRAIN_FILES = [f for f in os.listdir('train') if os.path.isfile(os.path.join('train', f))]
    print('--start--')
    await asyncio.gather(*[get_item(x, len(X)) for i, x in X.iterrows()])
    print('')
    print('--finish--')


def get_train_data(X):
    loop = asyncio.get_event_loop()
    async def main(X):
        return await asyncio.wait([get_for_items(X)])
    loop.run_until_complete(main(X))
    loop.close()


def get_test():
    loop = asyncio.get_event_loop()
    async def main():
        return await asyncio.wait([get_for_tests()])
    train_X = loop.run_until_complete(main())
    loop.close()


async def get_for_test_item(current_url, data=None, lvl=13, count=1):
    try:
        if data is None:
            columns = ['id', 'brand', 'color', 'fuelType', 'modelDate', 'numberOfDoors',
                       'productionDate', 'vehicleTransmission', 'enginePower', 'mileage',
                       'Привод', 'Руль', 'Состояние', 'Владельцы', 'ПТС', 'Таможня',
                       'Владение', 'motor', 'внедорожник', 'Безопасность', 'Салон',
                       'Мультимедиа', 'Комфорт', 'Обзор', 'Защита от угона', 'седан',
                       'хэтчбек', 'Элементы экстерьера', 'Прочее', 'лифтбек', 'купе', 'пикап',
                       'минивэн', 'компактвэн', 'универсал', 'родстер', 'кабриолет', 'фургон',
                       'микровэн', 'тарга', 'лимузин', 'model']
            data = pd.DataFrame([[0 for _ in range(0, len(columns))]], columns=columns).iloc[0]
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(data)
        new_data = None
        response = await get_page(current_url)
        if response:
            soup = BeautifulSoup(response, "html.parser")
            new_data = data.copy()
            new_data['car_id'] = int(re.search(r'\d+(?=\.html)', current_url).group())
            new_data['url'] = current_url

            for dv in soup.select_one('[data-bull-price]').findAll(text=True):
                price = re.sub(r'\s', '', dv)
                if price.isdigit():
                    new_data['price'] = int(price)
                    break

            for title in soup.select('h1', class_='b-title b-title_type_h1 b-title_no-margin b-text-inline'):
                for brand in BRAND:
                    if brand.lower() in title.text.strip().lower():
                        new_data['brand'] = BRAND.get(brand, 0)
                new_data['modelDate'] = int(re.search(r"\d{4}(?=\s+год)", title.text.strip().lower()).group() or 0)
            for description in soup.select('[data-section="auto-description"]'):
                for attribute in description.findAll('div', class_='b-media-cont_margin_b-size-xxs'):
                    textNodes = attribute.findAll(text=True)
                    if textNodes[0] == 'Двигатель:' and lvl >= 2:
                        new_data['fuelType'] = FUEL.get(textNodes[1].strip().replace(',', '').split()[0], 0)
                    elif textNodes[0] == 'Мощность:':
                        new_data['enginePower'] = int(
                            attribute.find(class_='b-triggers__text').text.replace('л.с.', '').replace(' ', ''))
                    elif textNodes[0] == 'Трансмиссия:' and lvl >= 5:
                        new_data.at['vehicleTransmission'] = {'механика': 1, 'автомат': 2}.get(textNodes[1].strip().split()[0], 0)
                    elif textNodes[0] == 'Привод:' and lvl >= 6:
                        new_data['Привод'] = WHEEL_DRIVE.get(textNodes[1].strip().split()[0], 0)
                    elif textNodes[0] == 'Тип кузова:' and lvl >= 11:
                        for b_name in BODY_TYPES:
                            new_data[b_name] = 0
                        bodyType = BODY_TYPES.get(textNodes[1].strip().replace(',', '').split()[0], [])
                        for bt in bodyType:
                            new_data[bt] = 1
                        if 'дв.' in textNodes[1]:
                            new_data['numberOfDoors'] = int(textNodes[1].strip().replace('дв.', '').strip()[-1])
                        else:
                            if FRAMETYPES.get(textNodes[1].strip().replace(',', '').split()[0]) in (10, 12):
                                new_data['numberOfDoors'] = 4
                            elif FRAMETYPES.get(textNodes[1].strip().replace(',', '').split()[0]) in (3, 6, 9):
                                new_data['numberOfDoors'] = 5
                            elif FRAMETYPES.get(textNodes[1].strip().replace(',', '').split()[0]) in (1, 11):
                                new_data['numberOfDoors'] = 2
                    elif textNodes[0] == 'Пробег, км:':
                        try:
                            new_data['mileage'] = round((int(textNodes[1].strip().split()[0].replace(',', '')) or 0) / 1.609344)
                        except ValueError:
                            new_data['mileage'] = 0
                        if new_data['mileage'] == 0 and lvl <= 7:
                            new_data['mileage'] = data['mileage']
                    elif textNodes[0] == 'Руль:' and lvl >= 2:
                        new_data['Руль'] = RORL.get(textNodes[1].strip().split()[0].title(), 0)
                    elif textNodes[0] == 'Особые отметки:' and lvl >= 9:
                        new_data['Состояние'] = 1 if textNodes[1].strip() == 'требуется ремонт или не на ходу' else 2
            productionDate = datetime.datetime.strptime(re.search(r'\d{2}-\d{2}-\d{4}', soup.select('[data-viewbull-views-counter]')[0].text.strip()).group(), '%d-%m-%Y').date()
            for description in soup.select('div[data-bull-id]'):
                mans_count = description.find(class_='b-flex bm-forceFlex b-text b-text_size_default')
                new_data['Владельцы'] = min(int(mans_count.findAll(class_='b-flex__item')[1].text), 3) if mans_count and mans_count.findAll(class_='b-flex__item')[0].text.strip() in ['Периоды регистрации', 'Записи о регистрации'] else 0

                time_long = description.find(
                        class_='b-media-cont b-media-cont_no-clear b-media-cont_bg_gray b-media-cont_modify_md b-random-group b-random-group_margin_b-size-xss b-text b-text_size_s'
                    )
                if time_long:
                    dates = time_long.find('div').findAll('div', class_='b-media-cont_margin_t-size-xxs')
                    if dates:
                        date = re.search(r'\d{2}.\d{2}.\d{4}', dates[-1].text)
                        new_data['Владение'] = (productionDate - datetime.datetime.strptime(
                            date.group(),
                            '%d.%m.%Y'
                        ).date()).days * 24 * 3600 if date else 0

            new_data['productionDate'] = productionDate.year
            new_data['ПТС'] = 2
            new_data['Таможня'] = 1
        elif count <= 10:
            new_data = await get_for_test_item(current_url, data, lvl, count+1)

        return new_data

    except Exception as error:
        print(f'{error=}')
        print(f'{current_url=}')
        print(traceback.print_tb(error.__traceback__))


def get_test_item():
    global TRAIN_FILES
    TRAIN_FILES = [f for f in os.listdir('train') if os.path.isfile(os.path.join('train', f))]
    loop = asyncio.get_event_loop()
    async def main():
        return await asyncio.wait([get_for_test_item('https://irkutsk.drom.ru/mercedes-benz/a-class/37626578.html')])
    train_X = loop.run_until_complete(main())
    loop.close()


def main(all_=True, new=True, train=True):
    df_train = pd.read_csv(DIR_TRAIN + 'test.csv')  # мой подготовленный датасет для обучения модели
    # df_test = pd.read_csv(DIR_TEST + 'test.csv')
    sample_submission = pd.read_csv(DIR_TEST + 'sample_submission.csv')

    train_preproc = preproc_data(df_train)
    # X_sub = preproc_data(sample_submission)

    # train_preproc.drop(['Привод', 'Руль', 'Владельцы', 'ПТС', 'enginePower'], axis=1, inplace=True, )  # убрал лишний столбец, которого нет в testе

    X = train_preproc

    get_train_data(X)

    for file in TRAIN_FILES:
        train_data = pd.read_csv(os.path.join('train', file))
        if len(train_data) > 40:
            train_data = train_data[train_data['Владельцы'] > 0]
            if len(train_data) > 40:
                print('>40', file)
                return
            elif len(train_data) < 10:
                print('<10 >40', file)
        if len(train_data) < 10:
            print('<10', file)

    X.drop(['model'], axis=1, inplace=True)

    y = sample_submission.price.values

    # get_train_data(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=VAL_SIZE, shuffle=True,
                                                        random_state=RANDOM_SEED)

    # X_train = pd.get_dummies(X_train, columns=['Привод', 'Руль', 'Владельцы', 'ПТС', 'vehicleTransmission', 'fuelType', 'color', 'brand'])


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
    print('------------------------------------------------')
    main(train=False, new=False)
    # get_test()
    # get_test_item()
