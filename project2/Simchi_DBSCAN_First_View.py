
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {'col1': [1, 3,5,7,9,11,13,15,55,56,57,58,59,60,61,62,100,102,104,106,108,110,112,114], 'col2': [100,102,104,106,108,110,112,114,66,65,64,63,62,61,60,59,1,3,5,7,9,11,13,15]}
df = pd.DataFrame(data=data)
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
        if distance < 10 :
            label.append(label[i])
        else :
            label.append(pre_l[0])
            pre_l.remove(pre_l[0])


            
df['Label'] = label
sns.scatterplot(data=df, x="col1", y="col2", hue="Label")
