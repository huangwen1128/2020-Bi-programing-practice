import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import gc
"""
user_log = pd.read_csv('./data_format1_small/sample_user_log.csv', dtype={'time_stamp':'str'})
user_info = pd.read_csv('./data_format1_small/sample_user_info.csv')
train_data1 = pd.read_csv('./data_format1_small/train.csv')
submission = pd.read_csv('./data_format1_small/test.csv')
"""
user_log = pd.read_csv('./data_format1/user_log_format1.csv', dtype={'time_stamp':'str'})
user_info = pd.read_csv('./data_format1/user_info_format1.csv')
train_data1 = pd.read_csv('./data_format1/train_format1.csv')
submission = pd.read_csv('./data_format1/test_format1.csv')
train_data2 = pd.read_csv('./data_format2/train_format2.csv')

train_data1['origin'] = 'train'
submission['origin'] = 'test'
print(user_log[:10])
print(user_log.info())
print(user_log.describe())
print(user_log.isnull().sum())
print(user_info[:10])
print(user_info.info())
print(user_info.describe())
print(user_info.isnull().sum())
#查看缺失值
print(user_log['brand_id'].value_counts())
print(user_info['age_range'].value_counts())
print(user_info['gender'].value_counts())
sns.countplot(data=train_data1, x='label')

matrix = pd.concat([train_data1, submission], ignore_index=True,  sort=False)
print(matrix[:10])
matrix.drop('prob', inplace=True, axis=1)
print(matrix[:10])

#matrix拼接用户数据
matrix = matrix.merge(user_info, on='user_id', how='left')
print(matrix)
#数据格式化和填充缺失值
user_log['time_stamp'] = pd.to_datetime(user_log['time_stamp'], format='%H%M')
user_log['brand_id'].fillna(0, inplace=True)
matrix['age_range'].fillna(0, inplace=True)
matrix['gender'].fillna(2, inplace=True)
del user_info, train_data1
gc.collect()
print(matrix)

#对user_log用户行为数据进行处理
user_log.rename(columns={'seller_id':'merchant_id'}, inplace=True)
groups = user_log.groupby(['user_id'])
temp = groups.size().reset_index().rename(columns={0:'u_record_nums'})
matrix = matrix.merge(temp, on='user_id', how='left')
#temp = user_log.groupby(['user_id'])['item_id'].agg({'item_count':'nunique'})
temp = groups['item_id'].agg([('item_count','nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['cat_id'].agg([('cat_count','nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['merchant_id'].agg([('merchant_count','nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')
temp = groups['brand_id'].agg([('merchant_count','nunique')]).reset_index()
matrix = matrix.merge(temp, on='user_id', how='left')

temp = groups['time_stamp'].agg([('F_time','max'),('L_time','min')]).reset_index()
temp['diff_F_L_time'] = (temp['F_time'] - temp['L_time']).dt.seconds/3600
matrix = matrix.merge(temp[['user_id','diff_F_L_time']], on='user_id', how='left')

#1表示添加到购物车，2表示购买，3表示添加到收藏夹
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'u_click_count',1:'u_cart_count',2:'u_buy_count',3:'u_collect_num'})
temp.fillna(0, inplace=True)
matrix = matrix.merge(temp, on='user_id', how='left')

#按照merchant_id统计随机负采样的个数
temp = train_data2[train_data2['label'] == -1].groupby(['merchant_id']).size().reset_index().rename(columns={0:'m_label_-1'})
matrix = matrix.merge(temp, on='merchant_id', how='left')
print(matrix)

#处理商家特征处理
groups = user_log.groupby(['merchant_id'])
#商家被交互行为的数量
temp = groups.size().reset_index().rename(columns={0:'m_record_nums'})
matrix = matrix.merge(temp, on='merchant_id', how='left')

#商家被交互的user_id,item_id,cat_id,brand_id 唯一值个数
temp = groups['user_id','item_id','cat_id','brand_id'].nunique().reset_index().rename(columns={'user_id':'m_user_count','item_id':'m_item_count','cat_id':'m_cat_count', 'brand_id':'m_brand_count'})
matrix = matrix.merge(temp, on='merchant_id', how='left')
#统计商家被交互的action_type唯一值数量
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'m_click_count',1:'m_cart_count',2:'m_buy_count',3:'m_collect_num'})
temp.fillna(0, inplace=True)
matrix = matrix.merge(temp, on='merchant_id', how='left')
print(matrix)

#按照user_id， merchant_id分组
groups = user_log.groupby(['user_id','merchant_id'])
temp = groups.size().reset_index().rename(columns={0:'um_record_count'})
matrix = matrix.merge(temp, on=['user_id','merchant_id'], how='left')

temp = groups['item_id','cat_id','brand_id'].nunique().reset_index().rename(columns={'item_id':'um_item_count','cat_id':'um_cat_count','brand_id':'um_brand_count'})
matrix = matrix.merge(temp, on=['user_id','merchant_id'], how='left')
#统计action_type
temp = groups['action_type'].value_counts().unstack().reset_index().rename(columns={0:'um_click_count',1:'um_cart_count',2:'um_buy_count',3:'um_collect_num'})
matrix = matrix.merge(temp, on=['user_id','merchant_id'], how='left')
temp = groups['time_stamp'].agg([('um_first','min'),('um_last','max')]).reset_index()
temp['um_diff'] = (temp['um_last'] - temp['um_first']).dt.seconds/3600
temp.drop(['um_first','um_last'], axis=1,inplace=True)
matrix = matrix.merge(temp, on=['user_id','merchant_id'], how='left')
print(matrix)

#用户购买点击比
matrix['u_buy_click_rate'] = matrix['u_buy_count']/matrix['u_click_count']
#商家购买点击比
matrix['m_buy_click_rate'] = matrix['m_buy_count']/matrix['m_click_count']
#不同用户不同商家购买点击比
matrix['um_buy_click_rate'] = matrix['um_buy_count']/matrix['um_click_count']
matrix.fillna(0, inplace=True)

temp = pd.get_dummies(matrix['age_range'], prefix='age')
matrix = pd.concat([matrix, temp], axis=1)
temp = pd.get_dummies(matrix['gender'], prefix='g')
matrix = pd.concat([matrix, temp], axis=1)
matrix.drop(['age_range','gender'], axis=1, inplace=True)
print(matrix)

#分割训练集数据和测试数据
train_data = matrix[matrix['origin'] == 'train'].drop(['origin'], axis=1)
test_data = matrix[matrix['origin'] == 'test'].drop(['origin','label'], axis=1)
train_X, train_y = train_data.drop(['label'],axis=1), train_data['label']
del temp, matrix
gc.collect

train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import xgboost as xgb

#将训练集进行切分
X_train, X_valid, y_train, y_valid = train_test_split(train_X, train_y, test_size=0.2)


#使用XGBoost
model = xgb.XGBClassifier(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=300,
    colsample_bytree=0.8,
    subsample=0.8,
    eta=0.3,
    seed=42,
    #scale_pos_weight=15
)
model.fit(
    X_train, y_train,
    eval_metric='auc', eval_set=[(X_train, y_train),(X_valid, y_valid)],
    verbose = True,
    early_stopping_rounds=10
)

prob = model.predict_proba(test_data)
submission['prob'] = pd.Series(prob[:,1])
submission.drop(['origin'], axis=1, inplace=True)
submission.to_csv('xgb_prediction.csv', index=False)

#K折
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
K_predict = pd.DataFrame()
i =0 
kf = KFold(n_splits=5)
temp = train_data.drop(['label'], axis=1)
for train_index, valid_index in kf.split(temp):
    X_new_train = temp.iloc[train_index]
    y_new_train = train_data['label'][train_index]
    X_new_valid = temp.iloc[valid_index]
    y_new_valid = train_data['label'][valid_index]

    model.fit(X_new_train, y_new_train,
        eval_metric='auc', eval_set=[(X_new_train, y_new_train),(X_new_valid, y_new_valid)],
        verbose = True,
        early_stopping_rounds=10)
    prob = model.predict_proba(test_data)
    K_predict[i] = pd.Series(prob[:,1])
    i +=1

submission['prob'] = K_predict.mean(axis=1)
submission.to_csv('xgb_prediction.csv', index=False)




