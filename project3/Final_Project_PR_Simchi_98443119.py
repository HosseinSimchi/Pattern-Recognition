import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier


#First Step (مرحله ي اول)

features = ['R_mercury','theta_mercury','R_venus','theta_venus','R_mars','theta_mars','R_jupiter','theta_jupiter','R_saturn'\
           ,'theta_saturn','R_uranus','theta_uranus','R_neptun','theta_neptun','R_moon','theta_moon']
df = pd.read_excel('C:\\Users\\Lenovo\\Desktop\\PR\\data_base_2.xlsx')
li = []
for i in range(df.shape[0]):
    r = df.iloc[i]['Mag']
    if r > 4.5 :
        li.append(1)
    else :
        li.append(0)
df['Label'] = li
kf = KFold(n_splits=5)
array = df.values
X = array[:,8:24]
y = array[:,24]
p = []
p1 = []
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    clf = GradientBoostingClassifier(random_state=0)
    #clf = SVC()
    #clf = RandomForestClassifier(max_depth=10, random_state=0)
    #clf = tree.DecisionTreeClassifier()
    clf.fit(Xtrain,ytrain)
    r = clf.predict(Xtest)
    plot_confusion_matrix(clf, Xtest, ytest)
    plt.show()
    u = accuracy_score(ytest, r)
    print(u)
    scores = cross_val_score(clf, Xtrain, ytrain, cv=5)
    e = precision_recall_fscore_support(ytest, r, average='macro')
    print(e)

#Second Step (Feature Selection)

kf = KFold(n_splits=5)
array = df.values
x = array[:,8:24]
y = array[:,24]
X = SelectKBest(chi2, k=10).fit_transform(x, y)
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
under = RandomUnderSampler(sampling_strategy=0.4)
X, y = under.fit_resample(X, y)
counter = Counter(y)
print(counter)
from imblearn.over_sampling import SMOTE
Oversampling = SMOTE(random_state=42)
X, y = Oversampling.fit_resample(X, y)
counter = Counter(y)
print(counter)
from sklearn.utils import shuffle
X, y = shuffle(X, y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
p = []
p1 = []
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    #clf = GradientBoostingClassifier(random_state=0)
    #clf = SVC()
    #clf = RandomForestClassifier(max_depth=10, random_state=0)
    clf = tree.DecisionTreeClassifier()
    clf.fit(Xtrain,ytrain)
    r = clf.predict(Xtest)
    plot_confusion_matrix(clf, Xtest, ytest)
    plt.show()
    u = accuracy_score(ytest, r)
    print(u)
    scores = cross_val_score(clf, Xtrain, ytrain, cv=5)
    e = precision_recall_fscore_support(ytest, r, average='macro')
    print(e)

#Third part

pca = PCA(n_components=10)
x = df.loc[:, features].values
y = df.loc[:,['Label']].values
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['0','1','2','3','4','5','6','7','8','9'])
finalDf = pd.concat([principalDf, df[['Label']]], axis = 1)
kf = KFold(n_splits=5)
array = finalDf.values
X = array[:,0:10]
y = array[:,10]
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
under = RandomUnderSampler(sampling_strategy=0.4)
X, y = under.fit_resample(X, y)
counter = Counter(y)
print(counter)
from imblearn.over_sampling import SMOTE
Oversampling = SMOTE(random_state=42)
X, y = Oversampling.fit_resample(X, y)
counter = Counter(y)
print(counter)
from sklearn.utils import shuffle
X, y = shuffle(X, y)
p = []
p1 = []
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    clf = GradientBoostingClassifier(random_state=0)
    #clf = SVC()
    #clf = RandomForestClassifier(max_depth=10, random_state=0)
    #clf = tree.DecisionTreeClassifier()
    clf.fit(Xtrain,ytrain)
    r = clf.predict(Xtest)
    plot_confusion_matrix(clf, Xtest, ytest)
    plt.show()
    u = accuracy_score(ytest, r)
    print(u)
    scores = cross_val_score(clf, Xtrain, ytrain, cv=5)
    e = precision_recall_fscore_support(ytest, r, average='macro')
    print(e)

#Forth part

li = []
for i in range(df.shape[0]):
    r = df.iloc[i]['Mag']
    if r > 4.5 :
        li.append(1)
    else :
        li.append(0)
df['Label'] = li

array = df.values
x = array[:,8:24]
y = array[:,24]
x = SelectKBest(chi2, k=12).fit_transform(x, y)
pca = PCA(n_components=10)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['0','1','2','3','4','5','6','7','8','9'])
finalDf = pd.concat([principalDf, df[['Label']]], axis = 1)
kf = KFold(n_splits=5)
array = finalDf.values
X = array[:,0:10]
y = array[:,10]
#X = SelectKBest(chi2, k=10).fit_transform(x, y)
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
under = RandomUnderSampler(sampling_strategy=0.4)
X, y = under.fit_resample(X, y)
counter = Counter(y)
print(counter)
from imblearn.over_sampling import SMOTE
Oversampling = SMOTE(random_state=42)
X, y = Oversampling.fit_resample(X, y)
counter = Counter(y)
print(counter)
from sklearn.utils import shuffle
X, y = shuffle(X, y)
p = []
p1 = []
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    clf = GradientBoostingClassifier(random_state=0)
    #clf = SVC()
    #clf = RandomForestClassifier(max_depth=10, random_state=0)
    #clf = tree.DecisionTreeClassifier()
    clf.fit(Xtrain,ytrain)
    r = clf.predict(Xtest)
    plot_confusion_matrix(clf, Xtest, ytest)
    plt.show()
    u = accuracy_score(ytest, r)
    print(u)
    scores = cross_val_score(clf, Xtrain, ytrain, cv=5)
    e = precision_recall_fscore_support(ytest, r, average='macro')
    print(e)

#Fifth Step

kf = KFold(n_splits=5)
array = df.values
x = array[:,8:24]
y = array[:,24]
X = SelectKBest(chi2, k=10).fit_transform(x, y)
X = StandardScaler().fit_transform(X)
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
under = RandomUnderSampler(sampling_strategy=0.2)
X, y = under.fit_resample(X, y)
counter = Counter(y)
print(counter)
from imblearn.over_sampling import SMOTE
Oversampling = SMOTE(random_state=42)
X, y = Oversampling.fit_resample(X, y)
counter = Counter(y)
print(counter)
from sklearn.utils import shuffle
X, y = shuffle(X, y)

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
p = []
p1 = []
for train_index, test_index in kf.split(X):
    Xtrain, Xtest = X[train_index], X[test_index]
    ytrain, ytest = y[train_index], y[test_index]
    clf = RandomForestClassifier(max_depth=10, random_state=0)
    clf.fit(Xtrain,ytrain)
    r = clf.predict(Xtest)
    plot_confusion_matrix(clf, Xtest, ytest)
    plt.show()
    u = accuracy_score(ytest, r)
    print(u)
    scores = cross_val_score(clf, Xtrain, ytrain, cv=5)
    e = precision_recall_fscore_support(ytest, r, average='macro')
    print(e)


#Predictions

df1 = pd.read_excel('C:\\Users\\Lenovo\\Desktop\\Book1.xlsx') 
print(df1)

array = df1.values
x1 = array[:,8:24]
y = [1,1,1,1]
X1 = SelectKBest(chi2, k=10).fit_transform(x1, y)
X1 = StandardScaler().fit_transform(X1)
r = clf.predict(X1)
r

