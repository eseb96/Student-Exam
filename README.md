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
![image](https://user-images.githubusercontent.com/64945381/110493144-25a0cb80-8125-11eb-9fa1-186c5ad29cc7.png)

Here we create a series of boxplots with the y-variable being the writing score and the x-variable being all the other parameters (except the __math score__ and __reading score__). From what can be seen the test preparation course and parental level of education plays the biggest part among all five of the parameters.

![image](https://user-images.githubusercontent.com/64945381/110494616-59c8bc00-8126-11eb-8bdc-de076d0970b9.png)
![image](https://user-images.githubusercontent.com/64945381/110494553-4e759080-8126-11eb-9f6d-8d5d2738d877.png)

A scatter plot is created to determine the relationship between the __reading score__ and __math score__ with the __writing score__. Here we can see a rather linear correlation for both the __reading score__ and __math score__. 
## 2. Data Preprocessing

## 3. Training the model

## 4. Conclusion
