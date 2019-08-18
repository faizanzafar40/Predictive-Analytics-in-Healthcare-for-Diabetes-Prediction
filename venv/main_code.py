import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
import pickle
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import numpy as np

SEED = 7

#import xgboost as xgb

np.random.seed(SEED)

warnings.filterwarnings('ignore')


data = pd.read_csv('..\input\diabetes.csv', engine='python')

#data2=pd.read_csv('C:\\Python36\FYP\input\diabetes.csv')

df_name=data.columns

#g = sns.PairGrid(data, vars=['Glucose', 'Insulin', 'BMI'], hue="Outcome", size=2.4)
#g.map_upper(plt.scatter)
#g.map_lower(sns.kdeplot, cmap="Blues_d")


#remove outliers of skin thickness

#max_skinthickness = data.SkinThickness.max()
#data = data[data.SkinThickness!=max_skinthickness]
#max_skinthickness = data2.SkinThickness.max()
#data2 = data2[data2.SkinThickness!=max_skinthickness]

def replace_zero(df, field, target):
    mean_by_target = df.loc[df[field] != 0, [field, target]].groupby(target).mean()
    data.loc[(df[field] == 0)&(df[target] == 0), field] = mean_by_target.iloc[0][0]
    data.loc[(df[field] == 0)&(df[target] == 1), field] = mean_by_target.iloc[1][0]

    # run function

for col in [ 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    replace_zero(data, col, 'Outcome')


for field in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'Age', 'DiabetesPedigreeFunction']:
            print('Field %s : num 0-entries: %d' % (field, len(data.loc[data[field] == 0, field])))


y = data.iloc[:, -1]

## ONE BY ONE

#X= data.iloc[:,:-1].drop('Glucose',1).drop('Pregnancies',1).drop('Insulin',1).drop('BMI',1).drop('BloodPressure',1).drop('DiabetesPedigreeFunction',1).drop('SkinThickness',1) ## f1score for Age only is 0.46
#X= data.iloc[:,:-1].drop('Glucose',1).drop('Pregnancies',1).drop('Age',1).drop('BMI',1).drop('BloodPressure',1).drop('DiabetesPedigreeFunction',1).drop('SkinThickness',1) #F1score for insulin only is 0.65
#X = data.iloc[:, :-1].drop('Insulin', 1).drop('Pregnancies', 1).drop('Age', 1).drop('BMI', 1).drop('BloodPressure',1).drop('DiabetesPedigreeFunction', 1).drop('SkinThickness', 1) #f1score for glucose is 0.57
#X= data.iloc[:,:-1].drop('Insulin',1).drop('Pregnancies',1).drop('Age',1).drop('Glucose',1).drop('BloodPressure',1).drop('DiabetesPedigreeFunction',1).drop('SkinThickness',1) #f1score for BMI is 0.293
#X= data.iloc[:,:-1].drop('Insulin',1).drop('Pregnancies',1).drop('Age',1).drop('BMI',1).drop('BloodPressure',1).drop('Glucose',1).drop('SkinThickness',1) #f1score for PedigreeFunction is 0.299
#X= data.iloc[:,:-1].drop('Insulin',1).drop('Pregnancies',1).drop('Age',1).drop('Glucose',1).drop('BloodPressure',1).drop('DiabetesPedigreeFunction',1).drop('BMI',1) # f1score for skinthickness is 0.463
#X= data.iloc[:,:-1].drop('Insulin',1).drop('Pregnancies',1).drop('Age',1).drop('Glucose',1).drop('BMI',1).drop('DiabetesPedigreeFunction',1).drop('SkinThickness',1) #f1score for BloodPressure is 0.126
#X= data.iloc[:,:-1].drop('Insulin',1).drop('BMI',1).drop('Age',1).drop('Glucose',1).drop('BloodPressure',1).drop('DiabetesPedigreeFunction',1).drop('SkinThickness',1) #f1score for pregnancies is 0.325


## 2 Fields

##X = data.iloc[:,:-1].drop('Pregnancies', 1).drop('BloodPressure', 1) gives 0.81 F1score
#X= data.iloc[:,:-1].drop('Age',1).drop('Pregnancies',1) gives 0.76 F1 score
#X= data.iloc[:,:-1].drop('Age',1).drop('BloodPressure',1) gives 0.76 F1 Score
##X= data.iloc[:,:-1].drop('BloodPressure',1).drop('BMI',1) gives 0.78 F1 Score

##X= data.iloc[:,:-1].drop('Glucose',1).drop('BMI',1) gives 0.76 f1score
##X= data.iloc[:,:-1].drop('DiabetesPedigreeFunction',1).drop('BMI',1)  gives 0.79 F1score
##X= data.iloc[:,:-1].drop('Pregnancies',1).drop('SkinThickness',1) gives 0.81 F1score
#X= data.iloc[:,:-1].drop('SkinThickness',1).drop('Age',1) gives 0.76 F1score
#X= data.iloc[:,:-1].drop('Glucose',1).drop('DiabetesPedigreeFunction',1) gives 0.78 f1score
#X= data.iloc[:,:-1].drop('BloodPressure',1).drop('BMI',1) gives 0.79 F1score
##X= data.iloc[:,:-1].drop('DiabetesPedigreeFunction',1).drop('Age',1) gives 0.78 f1score
#X = data.iloc[:,:-1].drop('BloodPressure',1) ## highest F1score of 0.82
#X = data.iloc[:,:-1].drop('Pregnancies', 1) ## F1 score of 0.80

X= data.iloc[:,:-1]
df_t = data.copy()
df_t_name = data.columns
df_name=data.columns
from sklearn.cross_validation import train_test_split

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.9, random_state=0) # 100 training set gives 0.36 f1score and 77% accuracy
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state=0) # 200 training set gives 0.55 f1score and 70% accuracy
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.7, random_state=0) # 300 training set gives 0.57 f1score and 75% accuracy
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.6, random_state=0) # 400 training set gives 0.49 f1score and 74% accuracy
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state=0) # 500 training set gives 0.51 f1score and 75% accuracy
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=0) # 600 training set gives 0.55 f1score and 75% accuracy
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=0) # 700 training set gives 0.36 f1score and 77% accuracy

"""
X_train= X_train.drop(X_train.tail(n=436).index) #accuracy for 100 train set is 87% and f1score is 0.81 model overfits
y_train= y_train.drop(y_train.tail(n=436).index)
"""
"""
X_train= X_train.drop(X_train.tail(n=336).index) #accuracy for 200 train set is 87% and f1score is 0.797 model overfits
y_train= y_train.drop(y_train.tail(n=336).index)
print(X_train.size)
"""
"""
X_train= X_train.drop(X_train.tail(n=236).index) #accuracy for 300 train set is 87% and f1score is 0.792 model overfits
y_train= y_train.drop(y_train.tail(n=236).index)
"""
"""
X_train= X_train.drop(X_train.tail(n=136).index) #accuracy for 400 train set is 88% and f1score is 0.819 Model doesnot overfit 
y_train= y_train.drop(y_train.tail(n=136).index)
"""

#X_test300= X_test.head(n=300)
#y_test300= y_test.head(n=300)
#X_train = X_train.append(X_train)
#y_train = y_train.append(y_train)

# add noise to a particular column
"""
column_to_add_noise = 'Glucose'
for i in range(0, len(X_train)):
    X_train.iloc[i, X_train.columns.get_loc(column_to_add_noise)] += random.randint(0,9)

# add noise to a particular column
column_to_add_noise = 'BloodPressure'
for i in range(0, len(X_train)):
    X_train.iloc[i, X_train.columns.get_loc(column_to_add_noise)] += random.randint(0,9)

# add noise to a particular column
column_to_add_noise = 'Age'
for i in range(0, len(X_train)):
    X_train.iloc[i, X_train.columns.get_loc(column_to_add_noise)] += random.randint(0,2)
"""


def TurkyOutliers(df_out,nameOfFeature,drop=False):

    valueOfFeature = df_out[nameOfFeature]

    # Calculate Q1 (25th percentile of the data) for the given feature

    Q1 = np.percentile(valueOfFeature, 25.)

    # Calculate Q3 (75th percentile of the data) for the given feature

    Q3 = np.percentile(valueOfFeature, 75.)

    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)

    step = (Q3-Q1)*1.5

    # print "Outlier step:", step

    outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].index.tolist()

    feature_outliers = valueOfFeature[~((valueOfFeature >= Q1 - step) & (valueOfFeature <= Q3 + step))].values

    # df[~((df[nameOfFeature] >= Q1 - step) & (df[nameOfFeature] <= Q3 + step))]


    # Remove the outliers, if any were specified

    print ("Number of outliers (inc duplicates): {} and outliers: {}".format(len(outliers), feature_outliers))
    if drop:
        good_data = df_out.drop(df_out.index[outliers]).reset_index(drop = True)
        print ("New dataset with removed outliers has {} samples with {} features each.".format(*good_data.shape))
        return good_data
    else:
        print ("Nothing happens, df.shape = ",df_out.shape)
        return df_out



feature_number=1
df_clean = TurkyOutliers(df_t,df_t_name[feature_number],True)


feature_number=2
df_clean = TurkyOutliers(df_clean,df_t_name[feature_number],True)

feature_number=3
df_clean = TurkyOutliers(df_clean,df_t_name[feature_number],True)

feature_number=4
df_clean = TurkyOutliers(df_clean,df_t_name[feature_number],True)

feature_number=5
df_clean = TurkyOutliers(df_clean,df_t_name[feature_number],True)

feature_number=6
df_clean = TurkyOutliers(df_clean,df_t_name[feature_number],True)

feature_number=7
df_clean = TurkyOutliers(df_clean,df_t_name[feature_number],True)


print('df shape: {}, new df shape: {}, we lost {} rows, {}% of our data'.format(df_t.shape[0],df_clean.shape[0],
                                                              df_t.shape[0]-df_clean.shape[0],
                                                        (df_t.shape[0]-df_clean.shape[0])/df_t.shape[0]*100))


X_clean= df_clean.iloc[:,:-1]
y_clean= df_clean.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size = 0.3, random_state=0)

#X_train= X_train.head(n=400)# at 50 f1 score is 0.77 but over fit, at 100 f1 score is 0.83 but overfit , at 150 f1 score is 0.844 but overfit decrease ,at 200 f1 score is 0.833 but overfit decreases,at 250 f1 score is 0.824 overfit decreases,at 300 f1 score is,at 300 f1 score is 0.851 and training set f1 score remains same,at 400 f1score is 0.846 and training set f1score remains decreases
#y_train= y_train.head(n=400)

print(X_train.shape)
print(X_test.shape)
print(y_train.size)
print(y_test.size)

# helper functions

def train_clf(clf, X_train, y_train):
    return clf.fit(X_train, y_train)

def pred_clf(clf, features, target):
    y_pred = clf.predict(features)
    return f1_score(target.values, y_pred, pos_label=1)

def accu_clf(clf, features, target):
    y_pred=clf.predict(features)
    return accuracy_score(target.values,y_pred)

def train_predict(clf, X_train, y_train, X_test, y_test):
    train_clf(clf, X_train, y_train)

    print("F1 score for training set is: {:.3f}".format(pred_clf(clf, X_train, y_train)))
    print("F1 score for testing set is: {:.3f}\n".format(pred_clf(clf, X_test, y_test)))
    print("Training Accuracy is {}".format(accu_clf(clf,X_train,y_train)))
    print("Testing Accuracy is {}\n".format(accu_clf(clf, X_test, y_test)))



# load algorithms

nb=GaussianNB()
knn = KNeighborsClassifier()
dtc = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)
abc = AdaBoostClassifier(random_state=0)
logreg = LogisticRegression()
algorithms = [logreg, knn, dtc, rfc, abc,nb]

for clf in algorithms:
 
    #print("\n{}: \n".format(clf.__class__.__name__))

    # create training data from first 100, then 200, then 300
    #for n in [179, 358, 537]:
        #train_predict(clf, X_train[:n], y_train[:n], X_test, y_test)
  
    print("{}:".format(clf))
    train_predict(clf, X_train, y_train, X_test, y_test)

##for n in range(3,10):
  ##  knn = KNeighborsClassifier(n_neighbors=n)
   # print("Number of neighbors is: {}".format(n))
    #train_predict(knn, X_train_cv, y_train_cv, X_test_cv, y_test_cv)

#Cross validation for best number of estimators

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
model= Sequential()
model.add(Dense(8, input_dim=8, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile the model

model.compile(loss='binary_crossentropy', optimizer='adagrad',metrics=['accuracy'])
model.summary()
model.fit(X_test, y_test, epochs=200, batch_size=50)

y_pred = model.predict(X_test)
print("{}:".format(model))
score = model.evaluate(X_test, y_test)
print("Accuracy --->", score[1])


#cross validation
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold

params = {'n_estimators': 1200, 'max_depth': 9, 'subsample': 0.5, 'learning_rate': 0.01, 'min_samples_leaf': 1,
          'random_state': 0}
gbc = GradientBoostingClassifier(**params)

n_estimators = 10
clf = gbc

# split training set into training and testing set
X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train, test_size=0.3, random_state=100)

X_train_cv = X_train_cv.reset_index(drop=True, inplace=False)
y_train_cv = y_train_cv.reset_index(drop=True, inplace=False)

clf.fit(X_train_cv, y_train_cv)
# score = f1_score(y_train, clf.predict(X_train), pos_label = 1)
acc = clf.score(X_test_cv, y_test_cv)

n_estimators = params['n_estimators']
x = np.arange(n_estimators) + 1


#The following code is credited solely to Sklearn documentation


def heldout_score(clf, X_test_cv, y_test_cv):
    #compute deviance scores on ``X_test`` and ``y_test``. 
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X_test_cv)):
        score[i] = clf.loss_(y_test_cv, y_pred)
    return score


def cv_estimate(n_splits=10):
    cv = KFold(n_splits=n_splits)
    cv_clf = clf
    val_scores = np.zeros((n_estimators,), dtype=np.float64)
    for train, test in cv.split(X_train_cv):
        cv_clf.fit(X_train_cv.iloc[train], y_train_cv[train])
        val_scores += heldout_score(cv_clf, X_train_cv.iloc[test], y_train_cv[test])
    val_scores /= n_splits
    return val_scores


# Estimate best n_estimator using cross-validation
cv_score = cv_estimate(3)

# Compute best n_estimator for test data
test_score = heldout_score(clf, X_test_cv, y_test_cv)

# negative cumulative sum of oob improvements
cumsum = -np.cumsum(clf.oob_improvement_)

# min loss according to OOB
oob_best_iter = x[np.argmin(cumsum)]

# min loss according to test (normalize such that first loss is 0)
test_score -= test_score[0]
test_best_iter = x[np.argmin(test_score)]

# min loss according to cv (normalize such that first loss is 0)
cv_score -= cv_score[0]
cv_best_iter = x[np.argmin(cv_score)]

# color brew for the three curves
oob_color = list(map(lambda x: x / 256.0, (190, 174, 212)))
test_color = list(map(lambda x: x / 256.0, (127, 201, 127)))
cv_color = list(map(lambda x: x / 256.0, (253, 192, 134)))

# plot curves and vertical lines for best iterations
plt.plot(x, cumsum, label='OOB loss', color=oob_color)
plt.plot(x, test_score, label='Test loss', color=test_color)
plt.plot(x, cv_score, label='CV loss', color=cv_color)
plt.axvline(x=oob_best_iter, color=oob_color)
plt.axvline(x=test_best_iter, color=test_color)
plt.axvline(x=cv_best_iter, color=cv_color)

# add three vertical lines to xticks
xticks = plt.xticks()
xticks_pos = np.array(xticks[0].tolist() +
                      [oob_best_iter, cv_best_iter, test_best_iter])
xticks_label = np.array(list(map(lambda t: int(t), xticks[0])) +
                        ['OOB', 'CV', 'Test'])
ind = np.argsort(xticks_pos)
xticks_pos = xticks_pos[ind]
xticks_label = xticks_label[ind]
plt.xticks(xticks_pos, xticks_label)

plt.legend(loc='upper right')
plt.ylabel('normalized loss')
plt.xlabel('number of iterations')

plt.show()


gbc = GradientBoostingClassifier(random_state=0,max_depth=3, subsample=0.5,learning_rate=0.01,n_estimators=310)
#gbc= GradientBoostingClassifier(random_state=0)
clf_ = gbc.fit(X_train, y_train)

y_pred = clf_.predict(X_test)
Gradientboost = open("gradientboost.pkl","wb")
pickle.dump(clf_,Gradientboost)
Gradientboost.close()
print("{}:".format(clf_))
print("F1 score for training set is: {:.3f}".format(pred_clf(clf_, X_train, y_train)))
print("Training Accuracy is {}\n".format(accu_clf(clf_,X_train,y_train)))
print("F1 score for testing set is: {:.3f}".format(pred_clf(clf_, X_test, y_test)))
print("Testing Accuracy is {}\n".format(accu_clf(clf_, X_test, y_test)))

print("\nImportance of Each feature")

print(gbc.feature_importances_)


"""

X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(X_train, y_train, test_size = 0.3, random_state=100)
for n in range(3,10):
    knn = KNeighborsClassifier(n_neighbors=n)
    print("Number of neighbors is: {}".format(n))
    train_predict(knn, X_train_cv, y_train_cv, X_test_cv, y_test_cv)


from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=8)
clf_ = knn.fit(X_train, y_train)
y_pred = clf_.predict(X_test)
print('Accuracy is {}'.format(accuracy_score(y_test,y_pred )))

"""

"""

def get_models():
    #Generate a library of base learners
    param = {'C': 0.7678243129497218, 'penalty': 'l1'}
    model1 = LogisticRegression(**param)

    param = {'n_neighbors': 15}
    model2 = KNeighborsClassifier(**param)

    param = {'C': 1.7, 'kernel': 'linear', 'probability':True}
    model3 = SVC(**param)

    param = {'criterion': 'gini', 'max_depth': 3, 'max_features': 2, 'min_samples_leaf': 3}
    model4 = DecisionTreeClassifier(**param)

    param = {'learning_rate': 0.05, 'n_estimators': 150}
    model5 = AdaBoostClassifier(**param)

    param = {'learning_rate': 0.01, 'n_estimators': 100}
    model6 = GradientBoostingClassifier(**param)

    model7 = GaussianNB()

    model8 = RandomForestClassifier()

    model9 = ExtraTreesClassifier()

    models = {'LR':model1, 'KNN':model2, 'SVC':model3,
              'DT':model4, 'ADa':model5, 'GB':model6,
              'NB':model7, 'RF':model8,  'ET':model9
              }

    return models
if  __name__ == "__main__":
    base_learners = get_models()
    meta_learner = GradientBoostingClassifier(
        n_estimators=1000,
        loss="exponential",
        max_features=6,
        max_depth=3,
        subsample=0.5,
        learning_rate=0.001,
        random_state=SEED
    )
    from mlens.ensemble import SuperLearner

    # Instantiate the ensemble with 10 folds
    sl = SuperLearner(
        folds=10,
        random_state=SEED,
        verbose=2,
        backend="multiprocessing"
    )

    # Add the base learners and the meta learner
    sl.add(list(base_learners.values()), proba=True)
    sl.add_meta(meta_learner, proba=True)

    # Train the ensemble
    sl.fit(X_train, y_train)

    # Predict the test set
    p_sl = sl.predict_proba(X_test)

    pp = []
    for p in p_sl[:, 1]:
        if p>0.5:
            pp.append(1.)
        else:
            pp.append(0.)

    print("\nSuper Learner Accuracy score: %.8f" % (y_test== pp).mean())

"""