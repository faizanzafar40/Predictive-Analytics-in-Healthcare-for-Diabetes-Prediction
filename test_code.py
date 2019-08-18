import seaborn as sb
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np

data = pd.read_csv('..\input\diabetes.csv')

y = data.iloc[:, -1]
X = data.iloc[:,:-1].drop('BloodPressure',1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

X_train = X_train.append(X_train)
y_train = y_train.append(y_train)

column_to_add_noise = 'Glucose'

for i in range(0, len(X_train)):
    X_train.iloc[i, X_train.columns.get_loc(column_to_add_noise)] += random.randint(0,9)

# add noise to a particular column

column_to_add_noise = 'Insulin'
for i in range(0, len(X_train)):
    X_train.iloc[i, X_train.columns.get_loc(column_to_add_noise)] += random.randint(0,9)

# add noise to a particular column

column_to_add_noise = 'Age'
for i in range(0, len(X_train)):
    X_train.iloc[i, X_train.columns.get_loc(column_to_add_noise)] += random.randint(0,2)

X_train = X_train.append(X_train)
y_train = y_train.append(y_train)

#apply algorithm

model = linear_model.LogisticRegression()

#fiting the model

model.fit(X_train, y_train)

#prediction

y_pred = model.predict(X_test)

#accuracy

print("Accuracy of Linear Regression-- >", model.score(X_test, y_test)*100)

#plot the confusion matrix

sb.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sb.heatmap(cm, annot=True, fmt='g')

#apply algorithm

model = RandomForestClassifier(n_estimators=1000)

#fiting the model

model.fit(X_train, y_train)

#prediction

y_pred = model.predict(X_test)

#accuracy

print("Accuracy of Random Forest Classifier -- >", model.score(X_test, y_test)*100)

#plot the confusion matrix

sb.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sb.heatmap(cm, annot=True, fmt='g')

#applying algorithm

model = SVC(gamma=0.01)

#fiting the model

model.fit(X_train, y_train)

#prediction

y_pred = model.predict(X_test)

#accuracy

print("Accuracy of SVM -- >", model.score(X_test, y_test)*100)

#plot the confusion matrix

sb.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sb.heatmap(cm, annot=True, fmt='g')


#applying algorithm

model = KNeighborsClassifier(n_neighbors=20)

#fiting the model

model.fit(X_train, y_train)

#prediction

y_pred = model.predict(X_test)

#accuracy

print("Accuracy of KNN -- >", model.score(X_test, y_test)*100)

#plot the confusion matrix

sb.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sb.heatmap(cm, annot=True, fmt='g')


#apply algorithm

model = GradientBoostingClassifier()

#fiting the model

model.fit(X_train, y_train)

#prediction

y_pred = model.predict(X_test)

#accuracy

print("Accuracy of Gradient Boosting Classifier -- >", model.score(X_test, y_test)*100)

#plot the confusion matrix

sb.set(font_scale=1.5)
cm = confusion_matrix(y_pred, y_test)
sb.heatmap(cm, annot=True, fmt='g')


class Metrics(Callback):
        def on_train_begin(self, logs={}):
                self.val_f1s = []
                self.val_recalls = []
                self.val_precisions = []


def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: % f — val_precision: % f — val_recall % f " ,_val_f1, _val_precision, _val_recall)
        return

metrics = Metrics()
from sklearn.neural_network import MLPClassifier
model = MLPClassifier(hidden_layer_sizes=(200,), max_iter=500, alpha=0.0001,
                     solver='', verbose=10,  random_state=0,tol=0.00000001,batch_size=100)


"""

model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

"""

#compile the model

"""

model.compile(loss='binary_crossentropy', optimizer='adagrad',metrics=['accuracy'])
model.summary()

"""

model.fit(X_train, y_train)
y_predict=model.predict(X_test)
print(accuracy_score(y_test, y_predict))

#print("Score of Neural Network--->", score[0])

