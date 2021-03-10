# Student-Exam
Predicting the writing score of students using XGBoost regression model

[Dataset obtained here](https://www.kaggle.com/spscientist/students-performance-in-exams) 

## 1. Analyzing the data
```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


dataset = pd.read_csv('StudentsPerformance.csv')
dataset.head()
``` 
![image](https://user-images.githubusercontent.com/64945381/110489786-40257580-8122-11eb-86e0-e83d849da1d2.png)

Here we are given a total of 8 parameters in the dataset. One row represents the profile of the student and the results they got. Our aim is to build a regression model that can predict the __writing score__. To do that we have to first analyze whether or not each of the parameters correlate with one another and how signficant are they in predicting the __writing score__.

```python
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

Based on the image we can see all the parameters are correlated with the __writing score__ with the biggest correlation being the __math score__ and __reading score__. Our decision is to use all the parameters since we believe the lowest correlation 9% is still useful for creating our model.
## 2. Data Preprocessing
There were no NaN values in the dataset so we can skip data cleansing
```python
X_writing.head()
```
![image](https://user-images.githubusercontent.com/64945381/110572754-a810a600-818c-11eb-986a-0c52da6ce904.png)
The __X_writing__ variable stores the independant variables that will be used to dermine the __writing score__.
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse = False), list(range(5)))], remainder='passthrough')
X_writing = np.array(ct.fit_transform(X_writing))
```
Here we encode the string based categorical parameters into 1s and 0s for the machine learning algorithm to be able to process the parameters properly. 

## 3. Training the model

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_writing,y_writing,test_size=0.2,random_state=1)
```
We split our data to have a test size of 20% of the dataset. 

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,-2:]=sc.fit_transform(X_train[:,-2:])
X_test[:,-2:]=sc.transform(X_test[:,-2:])
```

We also want to normalize our data for the __math score__ and __reading score__ parameters by including a feature scaling. 

```python
print(X_train[0])
```
![image](https://user-images.githubusercontent.com/64945381/110576195-e90bb900-8192-11eb-91e9-4c5483de8e39.png)


The output shows the first column entry that has been encoded and feature scaled.

```python
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X_train,y_train)
```
![image](https://user-images.githubusercontent.com/64945381/110576319-296b3700-8193-11eb-85f8-bf04194cfe05.png)

There are several parameters usable for the XGBRegressor() method but we stick to our default parameters for now.

```python
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))
```
![image](https://user-images.githubusercontent.com/64945381/110576546-8bc43780-8193-11eb-8267-bd439db6c705.png)

Here is the result of the model against the first few test set entries, based on the results it seeems that the model is rather accurate at predicting the scores

```python
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
```
![image](https://user-images.githubusercontent.com/64945381/110576714-ca59f200-8193-11eb-99df-232c63e64307.png)

```python
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
```
![image](https://user-images.githubusercontent.com/64945381/110576738-da71d180-8193-11eb-863e-f74af7429085.png)

From the r2 score and cross valuation score the accuracy of the data is above 90%. So we can accept the model with a 90% confidence level
## 4. Conclusion

