# Import necessary modules 
import pandas as pd
import scipy.stats as st
import numpy as np
import seaborn as sns
from sklearn import datasets 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

iris = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\iris.csv")

iris['sepal length'] = iris['sepal length'].apply(lambda x: x*10)
iris['sepal width'] = iris['sepal width'].apply(lambda x: x*10)

for i in range(150):
    for j in range(2):
        if 1 <= iris.iloc[i, j] < 5:
            iris.iloc[i, j] = 1
        if 5 <= iris.iloc[i, j] < 10:
            iris.iloc[i, j] = 2
        if 10 <= iris.iloc[i, j] < 15:
            iris.iloc[i, j] = 3
        if 15 <= iris.iloc[i, j] < 20:
            iris.iloc[i, j] = 4
        if 20 <= iris.iloc[i, j] < 25:
            iris.iloc[i, j] = 5
        if 25 <= iris.iloc[i, j] < 30:
            iris.iloc[i, j] = 6
        if 30 <= iris.iloc[i, j] < 35:
            iris.iloc[i, j] = 7
        if 35 <= iris.iloc[i, j] < 40:
            iris.iloc[i, j] = 8
        if 40 <= iris.iloc[i, j] < 45:
            iris.iloc[i, j] = 9
        if 45 <= iris.iloc[i, j] < 50:
            iris.iloc[i, j] = 10
        if 50 <= iris.iloc[i, j] < 55:
            iris.iloc[i, j] = 11
        if 55 <= iris.iloc[i, j] < 60:
            iris.iloc[i, j] = 12
        if 60 <= iris.iloc[i, j] < 65:
            iris.iloc[i, j] = 13
        if 65 <= iris.iloc[i, j] < 70:
            iris.iloc[i, j] = 14
        if 70 <= iris.iloc[i, j] < 75:
            iris.iloc[i, j] = 15
        if 75 <= iris.iloc[i, j] < 80:
            iris.iloc[i, j] = 16
        if 85 <= iris.iloc[i, j] < 90:
            iris.iloc[i, j] = 17
        if 90 <= iris.iloc[i, j] < 95:
            iris.iloc[i, j] = 18
        if 95 <= iris.iloc[i, j] <= 100:
            iris.iloc[i, j] = 19


def plot_2d_kde(df):
    # Extract x and y
    x = df['sepal length']
    y = df['sepal width']
    # Define the borders
    deltaX = (max(x) - min(x))/10
    deltaY = (max(y) - min(y))/10
    xmin = min(x) - deltaX
    xmax = max(x) + deltaX
    ymin = min(y) - deltaY
    ymax = max(y) + deltaY

    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    # We will fit a gaussian kernel using the scipy’s gaussian_kde method
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(yy, xx, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.set_title('Surface plot of Gaussian 2D KDE')
    fig.colorbar(surf, shrink=0.5, aspect=5) # add color bar indicating the PDF
    ax.view_init(60, 35)
    plt.show()
plot_2d_kde(iris)

   
iris = datasets.load_iris() 
  
iris_df = pd.DataFrame(iris.data, columns=['Sepal_Length', 
                      'Sepal_Width', 'Patal_Length', 'Petal_Width']) 
  
iris_df['Target'] = iris.target 
  
iris_df['Target'].replace([0], 'Iris_Setosa', inplace=True) 
iris_df['Target'].replace([1], 'Iris_Vercicolor', inplace=True) 
iris_df['Target'].replace([2], 'Iris_Virginica', inplace=True) 
  

sns.kdeplot(iris_df.loc[(iris_df['Target']=='Iris_Setosa'), 
            'Sepal_Length'], color='r', shade=True, Label='Iris_Setosa') 
  
sns.kdeplot(iris_df.loc[(iris_df['Target']=='Iris_Virginica'),  
            'Sepal_Length'], color='b', shade=True, Label='Iris_Virginica') 
sns.kdeplot(iris_df.loc[(iris_df['Target']=='Iris_Vercicolor'),  
            'Sepal_Length'], color='black', shade=True, Label='Vercicolor') 

plt.xlabel('Sepal Length') 
plt.ylabel('Probability Density') 
plt.show()
iris = pd.read_csv("C:\\Users\\Lenovo\\Desktop\\iris.csv")
iris['sepal length'] = iris['sepal length'].apply(lambda x: x*10)
iris['sepal width'] = iris['sepal width'].apply(lambda x: x*10)

for i in range(150):
    for j in range(2):
        if 1 <= iris.iloc[i, j] < 5:
            iris.iloc[i, j] = 1
        if 5 <= iris.iloc[i, j] < 10:
            iris.iloc[i, j] = 2
        if 10 <= iris.iloc[i, j] < 15:
            iris.iloc[i, j] = 3
        if 15 <= iris.iloc[i, j] < 20:
            iris.iloc[i, j] = 4
        if 20 <= iris.iloc[i, j] < 25:
            iris.iloc[i, j] = 5
        if 25 <= iris.iloc[i, j] < 30:
            iris.iloc[i, j] = 6
        if 30 <= iris.iloc[i, j] < 35:
            iris.iloc[i, j] = 7
        if 35 <= iris.iloc[i, j] < 40:
            iris.iloc[i, j] = 8
        if 40 <= iris.iloc[i, j] < 45:
            iris.iloc[i, j] = 9
        if 45 <= iris.iloc[i, j] < 50:
            iris.iloc[i, j] = 10
        if 50 <= iris.iloc[i, j] < 55:
            iris.iloc[i, j] = 11
        if 55 <= iris.iloc[i, j] < 60:
            iris.iloc[i, j] = 12
        if 60 <= iris.iloc[i, j] < 65:
            iris.iloc[i, j] = 13
        if 65 <= iris.iloc[i, j] < 70:
            iris.iloc[i, j] = 14
        if 70 <= iris.iloc[i, j] < 75:
            iris.iloc[i, j] = 15
        if 75 <= iris.iloc[i, j] < 80:
            iris.iloc[i, j] = 16
        if 85 <= iris.iloc[i, j] < 90:
            iris.iloc[i, j] = 17
        if 90 <= iris.iloc[i, j] < 95:
            iris.iloc[i, j] = 18
        if 95 <= iris.iloc[i, j] <= 100:
            iris.iloc[i, j] = 19
X = iris.loc[:,:].values
x = X[:,:2]
y = X[:,2]
le = LabelEncoder()
y = le.fit_transform(y)
clf = GaussianNB()
scores = cross_val_score(clf, x, y, cv=20)

t1 = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']
plt.bar(t1, scores)
plt.xlabel('Bin numbers') 
plt.ylabel('Accuracy') 
plt.show()
