__author__ = 'p_cohen'

############################ Import packages ############################
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import json
import os

############################ Define globals ############################
RAW_PATH = '/home/vagrant/Coupon/Data/Raw/'
CLEAN_PATH = '/home/vagrant/Coupon/Data/Clean/'

############################ Define functions ############################
def encode_text(data):
    # by 'prasanna23'
    # This function converts the data into type 'str'.
    try:
        text = encode_text(data)
    except:
        text = data.encode("UTF-8")
    return text

def translate_file(data, word_map):
    """
    Replaces values in data according to a word translation map

    :param data: data which may have untranslated characters
    :param word_map: (dict) key=foriegn character, value=english equivalent
    :return: translated data
    """
    for col in data.columns:
        if data[col].dtype == 'object':
            for key, val in word_map.iteritems():
                data[col][data[col] == key] = val
    return data

def encode_string_features(d_frame, string_feats):
    """
    This builds features for the Kaggle Japanese coupon comp
    :param data: dataframe
    :return: dataframe with features
    """
    for feat in string_feats:
        nm = 'le_' + feat
        le = preprocessing.LabelEncoder()
        d_frame[nm] = le.fit_transform(d_frame[feat])
    return d_frame



############################ Execute ############################
## Translate areas
# Load conversion map
json_data = open(CLEAN_PATH + "translation.json").read()
translations = json.loads(json_data)

# Translate all files
for f in ['coupon_visit_test.csv',]: #os.listdir(RAW_PATH)
    try:
        df_totrans = pd.read_csv(RAW_PATH + f)
        df_totrans = translate_file(df_totrans, translations)
        df_totrans.to_csv(CLEAN_PATH + f, index=False)
    except pd.parser.CParserError:
        continue

# initialize list of features
Xfeats = []
# Load files
visit = pd.read_csv(CLEAN_PATH + 'coupon_visit_train.csv')
list = pd.read_csv(CLEAN_PATH + 'coupon_list_train.csv')
user = pd.read_csv(CLEAN_PATH + 'user_list.csv')
test_list = pd.read_csv(CLEAN_PATH + 'coupon_list_test.csv')

# Clean list
list_string_feats = ['CAPSULE_TEXT', 'GENRE_NAME', 'large_area_name',
                     'ken_name', 'small_area_name']
list['is_train'] = 1
test_list['is_train'] = 0
all_list = pd.append(list, test_list, ignore_index=True)
all_list = encode_string_features(all_list, list_string_feats)
tmp = [Xfeats.append('le_' + x) for x in list_string_feats]

# Clean user
user_string_feats = ['REG_DATE', 'SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME']
user = encode_string_features(user, user_string_feats)
tmp = [Xfeats.append('le_' + x) for x in user_string_feats]

# Construct train
train = pd.merge(visit, list, left_on="VIEW_COUPON_ID_hash",
                 right_on="COUPON_ID_hash")
train = pd.merge(train, user, left_on="USER_ID_hash", right_on="USER_ID_hash")


# clean test
test_list['one'] = 1
user['one'] = 1
test = pd.merge(test_list, user, on='one')


# Create features
string_feats = ['CAPSULE_TEXT', 'GENRE_NAME',
                'large_area_name', 'ken_name', 'small_area_name',
                'REG_DATE', 'SEX_ID', 'AGE', 'WITHDRAW_DATE', 'PREF_NAME']
num_feats = ['PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'USABLE_DATE_MON',
             'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
             'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN',
             'USABLE_DATE_HOLIDAY']


# Append numeric feat lists
tmp = [Xfeats.append(x) for x in num_feats] # Store None's in tmp
# Clean features
for feat in Xfeats:
    train[feat] = train[feat].fillna(-1)

forest = RandomForestClassifier(n_estimators=30, n_jobs=2)
forest.fit(train[Xfeats], train.PURCHASE_FLG)
outputs = pd.DataFrame({'feats': Xfeats,
                        'weight': forest.feature_importances_})
outputs = outputs.sort(columns='weight', ascending=False)
print outputs


test = create_features(test, string_feats)
# Clean features
for feat in Xfeats:
    test[feat] = test[feat].fillna(-1)
test["preds"] = forest.predict(test[Xfeats])
