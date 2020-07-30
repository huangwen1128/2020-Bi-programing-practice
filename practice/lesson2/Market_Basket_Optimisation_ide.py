# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd

data = pd.read_csv('practice/lesson2/Market_Basket_Optimisation.csv',header=None)
print(data.head())
data['ColumnA'] = data[data.columns[0:]].apply(
    lambda x: ','.join(x.dropna().astype(str)),
    axis=1
)
data_merg =  pd.DataFrame(data=data['ColumnA'].values, columns=['ColumnA'])
data_hot_encoded = data_merg.drop('ColumnA', 1).join(data_merg['ColumnA'].str.get_dummies(','))
print(data_hot_encoded.head())

print(data_hot_encoded.sum().sort_values(ascending = False)[:10])
from nltk.tokenize import word_tokenize
'''
#可视化展示
import seaborn as sns
import matplotlib.pylab as plt
corr_data = data_hot_encoded.corr()
sns.heatmap(corr_data)
plt.show()
'''
all_word = " ".join(data[data.columns[0:]].apply(lambda x: " ".join(x.dropna().astype(str)),axis=1))

def remove_stop_words(f):
	stop_words = ['Movie']
	for stop_word in stop_words:
		f = f.replace(stop_word, '')
	return f

cut_text = word_tokenize(all_word)
#print(cut_text)
cut_text = " ".join(cut_text)
from wordcloud import WordCloud
wc = WordCloud(
		max_words=10,
		width=2000,
		height=1200,
        collocations=False,
)
wordcloud = wc.generate(cut_text)
# 写词云图片
wordcloud.to_file("wordcloud.jpg")
# 显示词云文件
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
process_word = WordCloud.process_text(wc, cut_text)
sorted(process_word.items(),key=lambda item:item[1],reverse=True)