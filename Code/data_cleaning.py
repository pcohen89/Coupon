__author__ = 'p_cohen'

############################ Import packages ############################
import pandas as pd
from sklearn import preprocessing
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

def create_features(d_frame, string_feats):
    """
    This builds features for the Kaggle Japanese coupon comp
    :param data: dataframe
    :return: dataframe with features
    """
    d_frame['loc_match'] = d_frame.PREF_NAME == d_frame.ken_name
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

# Merge files
visit = pd.read_csv(CLEAN_PATH + 'coupon_visit_train.csv')
list = pd.read_csv(CLEAN_PATH + 'coupon_list_train.csv')
user = pd.read_csv(CLEAN_PATH + 'user_list.csv')

train = pd.merge(visit, list, left_on="VIEW_COUPON_ID_hash",
                 right_on="COUPON_ID_hash")
train = pd.merge(train, user, left_on="USER_ID_hash", right_on="USER_ID_hash")

# Create features
string_feats = [u'PAGE_SERIAL', u'USER_ID_hash', u'CAPSULE_TEXT', u'GENRE_NAME',
                u'PRICE_RATE', u'CATALOG_PRICE', u'DISCOUNT_PRICE',
                u'large_area_name', u'ken_name', u'small_area_name',
                u'REG_DATE', u'SEX_ID', u'AGE', u'WITHDRAW_DATE', u'PREF_NAME']
thingy = create_features(train, string_feats)