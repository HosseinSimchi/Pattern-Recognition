import pandas as pd
import numpy as np
from sklearn import preprocessing
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:\\Users\\Lenovo\\Desktop\\DBSCAN-master\\clusters3.csv')  

def radius(x,y):
    a = x[0]
    b = x[1]
    c = y[0]
    d = y[1]
    distance = np.sqrt((a-c)**2 + (b-d)**2)
    return distance


points=[]
for i in range(len(df['col1'])):
    points.append(list(df.iloc[i,:]))

label = ['A']
pre_l = ['B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R']

for i in range(len(points)):      
    if (i+1) != len(points):
        x = points[i]
        y = points[i+1]
        distance = radius(x,y)
        if (distance < 70) :#56
            label.append(label[i])
        else:
            label.append(pre_l[0])
            pre_l.remove(pre_l[0]) 

df['Label'] = label

sns.scatterplot(data=df, x="col1", y="col2", hue="Label")
