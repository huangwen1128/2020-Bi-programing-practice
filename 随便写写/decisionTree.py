from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
digits = load_digits()
data = digits.data
print(data.shape)
print(digits.images[0])
print(digits.target[0])

train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size=0.25)
ss =StandardScaler()
train_x = ss.fit_transform(train_x)
test_x = ss.fit_transform(test_x)
model = DecisionTreeClassifier()
model.fit(train_x, train_y)
predict_y = model.predict(test_x)
print('决策树准确率: %0.4lf' %accuracy_score(test_y, predict_y))

'''
# 创建SVM分类器
from sklearn.svm import SVC
model = SVC()
model.fit(train_x, train_y)
predict_y=model.predict(test_x)
print('SVC准确率: %0.4lf' % accuracy_score(predict_y, test_y))
'''
import pydotplus
from sklearn.externals.six import StringIO
from sklearn.tree import export_graphviz

def show_tree(clf):
    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("hh.pdf")
show_tree(model)