# -*- coding: utf-8 -*-
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd
N=500
x = np.random.randn(N)
y = np.random.randn(N)

#plt绘制散点图
plt.scatter(x, y, marker='x')
plt.show()

#sns绘制散点图
df = pd.DataFrame({'x':x, 'y':y})
sns.jointplot(x='x',y='y',data = df, kind='scatter')
plt.show()

x = [1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910]
y = [265, 323, 136, 220, 305, 350, 419, 450, 560, 720, 830]
#plt折线图
plt.plot(x,y)
plt.title('jjjj')
plt.xlabel("数量")
plt.ylabel('年份')
plt.show()

#sns折线图
df = pd.DataFrame({'数量':x, "年份":y})
df1 = df.copy()
df1['数量'] = df1['数量'] + np.random.random(1)[0]
sns.lineplot(x='数量',y='年份',data=df)
sns.lineplot(x='数量',y='年份',data=df1)
plt.title("spring")
plt.legend(["0","1"])
plt.show()

x = ['c1', 'c2', 'c3', 'c4']
y = [15, 18, 5, 26]
sex = [1,2,3,4]
#plt条形图
plt.bar(x,y)
plt.show()
#sns条形图
df = pd.DataFrame({"季度":x, "数量":y,"sex":sex})
sns.barplot(x='季度',y='数量',data=df)
plt.show()

data = np.random.normal(size=(10,4))
labels = ['A','B','C','D']
#plt箱型
plt.boxplot(data, labels=labels)
plt.show()
#sns箱体
df = pd.DataFrame(data, columns=labels)
sns.boxplot(data=df)
plt.show()

#饼图
nums = [23,45,22]
labels = ['a','b','c']
plt.pie(x=nums, labels=labels)
plt.show()

#热力图
np.random.seed(33)
data = np.random.randn(3,3)
sns.heatmap(data)
plt.show()

#蜘蛛图
labels = np.array(["推进","KDA","生存","团战","发育","输出"])
stats=[76, 58, 67, 97, 86, 58]
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
print(angles)
stats=np.concatenate((stats,[stats[0]]))
angles=np.concatenate((angles,[angles[0]]))
# 用Matplotlib画蜘蛛图
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)   
ax.plot(angles, stats, 'o-', linewidth=2)
ax.fill(angles, stats, alpha=0.25)

# 设置中文字体
#font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=14)  
ax.set_thetagrids(angles * 180/np.pi, labels)#, FontProperties=font)
plt.show()

