import abc
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, f1_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class BaseModel:

    def __init__(self, model, name):
        self.model = model
        self.name = name

    def classification_model_stats(self, x_train, x_test, y_train, y_test):
        model = self.model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_pred, y_test)
        recall= recall_score(y_pred, y_test)
        confusion= confusion_matrix(y_test, y_pred)
        f1 =f1_score(y_test, y_pred)
        print('Model used:' ,self.name)
        print('{} Accuracy:{}%' .format(self.name, accuracy*100))
        print('{} recall:{}%'.format(self.name, recall*100))
        print('{} Confusion Matrix:\n{}'.format(self.name, confusion))
        print('{} F1 score:{}%'.format(self.name, f1*100))
        return accuracy, recall, f1, y_pred

    def regression_model_stats(self, x_train, x_test, y_train, y_test):
        model = self.model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        mae = mean_absolute_error(y_pred, y_test)
        mse= mean_squared_error(y_pred, y_test)
        rmse= mse ** 0.5
        r2 =r2_score(y_test, y_pred)
        print('Model used:' ,self.name)
        print('{} MAE:{}%' .format(self.name, mae))
        print('{} MSE:{}%'.format(self.name, mse))
        print('{} RMSE:{}'.format(self.name, rmse))
        print('{} R Squared:{}%'.format(self.name,r2))
        return mae, mse, rmse, r2, y_pred

    @abc.abstractmethod
    def predict(self, X):
        pass

