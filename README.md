# Student-Exam
Predicting the writing score of students using XGBoost regression model

[Dataset obtained here](https://www.kaggle.com/spscientist/students-performance-in-exams) 

## 1. Analyzing the data
``` 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv('StudentsPerformance.csv')
dataset.head()
``` 
![image](https://user-images.githubusercontent.com/64945381/110489786-40257580-8122-11eb-86e0-e83d849da1d2.png)

Here we are given a total of 8 parameters in the dataset. One row represents the profile of the student and the results they got. Our aim is to build a regression model that can predict the __writing score__. To do that we have to first analyze whether or not each of the parameters correlate with one another and how signficant are they in predicting the __writing score__.

```
import seaborn as sns

fig,ax = plt.subplots(ncols=3,figsize=(20,5))
sns.boxplot(x='gender',y='writing score',data=dataset,ax=ax[0])
sns.boxplot(x='race/ethnicity',y='writing score',data=dataset,ax=ax[1])
sns.boxplot(x='test preparation course',y='writing score',data=dataset,ax=ax[2])
fig,ax = plt.subplots(ncols=2,figsize=(25,5))
sns.boxplot(x='lunch',y='writing score',data=dataset,ax=ax[0])
sns.boxplot(x='parental level of education',y='writing score',data=dataset,ax=ax[1])
```
![image](https://user-images.githubusercontent.com/64945381/110570820-210dfe80-8189-11eb-9409-0700c0c0b25e.png)

Here we create a series of boxplots with the y-variable being the writing score and the x-variable being all the other parameters (except the __math score__ and __reading score__). From what can be seen the test preparation course and parental level of education plays the biggest part among all five of the parameters.

![image](https://user-images.githubusercontent.com/64945381/110570854-2e2aed80-8189-11eb-9b8e-b37f77b2aaf2.png)

![image](https://user-images.githubusercontent.com/64945381/110570874-371bbf00-8189-11eb-901f-166f54cd4acd.png)


A scatter plot is created to determine the relationship between the __reading score__ and __math score__ with the __writing score__. Here we can see a rather linear correlation for both the __reading score__ and __math score__. 
```python
from sklearn import preprocessing

label = preprocessing.LabelEncoder()
df_categorical_encoded = pd.DataFrame()
for i in dataset.columns:
    df_categorical_encoded[i]=label.fit_transform(dataset[i])

from scipy.stats import chi2_contingency

def cramers_V(var1,var2):
    crosstab = np.array(pd.crosstab(var1,var2,rownames=None,colnames=None))
    stat = chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab)
    mini = min(crosstab.shape)-1
    return(stat/(obs*mini))
rows =[]

for var1 in df_categorical_encoded:
    col = []
    for var2 in df_categorical_encoded:
        cramers = cramers_V(df_categorical_encoded[var1],df_categorical_encoded[var2])
        col.append(round(cramers,2))
    rows.append(col)

cramers_results = np.array(rows)
cramerv_matrix = pd.DataFrame(cramers_results, columns=df_categorical_encoded.columns,index=df_categorical_encoded.columns)
mask=np.triu(np.ones_like(cramerv_matrix,dtype=np.bool))
cat_heatmap = sns.heatmap(cramerv_matrix,mask=mask,vmin=-1,vmax=1,annot=True,cmap='BrBG')
cat_heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12},pad=12);
```
![image](https://user-images.githubusercontent.com/64945381/110572125-6a5f4d80-818b-11eb-927a-a60bb98c7202.png)

## 2. Data Preprocessing

## 3. Training the model

## 4. Conclusion
