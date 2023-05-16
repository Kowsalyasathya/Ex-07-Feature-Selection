# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file
# CODE
Name : Kowsalya M
Register No: 212222230069
```
import pandas as pd
import numpy as np
import seaborn as sns
df=pd.read_csv('/content/titanic_dataset.csv')
df

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)
df.drop('Name',axis=1,inplace=True)
df.drop('Ticket',axis=1,inplace=True)
df.drop('PassengerId',axis=1,inplace=True)
df.drop('Parch',axis=1,inplace=True)
df.head()
df['Age']=df['Age'].fillna(df['Age'].median())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()

import numpy as np
import matplotlib.pyplot as plt
plt.title("Dataset with outliers")
df.boxplot()

plt.show()
cols = ['Age','SibSp','Fare']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()
from sklearn.preprocessing import OrdinalEncoder
climate = ['male','female']
en= OrdinalEncoder(categories = [climate])
df['Sex']=en.fit_transform(df[["Sex"]])
df.head()
from sklearn.preprocessing import RobustScaler
sc=RobustScaler()
df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])
df.head()
import statsmodels.api as sm
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()
df1["Survived"]=np.sqrt(df["Survived"])
df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])
df1["Sex"]=np.sqrt(df["Sex"])
df1["Age"]=df["Age"]
df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])
df1["Fare"],parameters=stats.yeojohnson(df["Fare"])
df1["Embarked"]=df["Embarked"]
df1.skew()
import matplotlib
import seaborn as sns
import statsmodels.api as sm
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
X = df1.drop("Survived",1) 
y = df1["Survived"]          

plt.figure(figsize=(7,6))
cor = df1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)
plt.show()
cor_target = abs(cor["Survived"])
relevant_features = cor_target[cor_target>0.5]
relevant_features
cols = list(X.columns)
pmax = 1
while (len(cols)>0):
    p= []
    X_1 = X[cols]
    X_1 = sm.add_constant(X_1)
    model = sm.OLS(y,X_1).fit()
    p = pd.Series(model.pvalues.values[1:],index = cols)      
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if(pmax>0.05):
        cols.remove(feature_with_p_max)
    else:
        break
selected_features_BE = cols
print(selected_features_BE)
model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)
print(rfe.support_)
print(rfe.ranking_)

nof_list=np.arange(1,6)            
high_score=0
nof=0           
score_list =[]
for n in range(len(nof_list)):
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    model = LinearRegression()
    rfe = RFE(model,step=nof_list[n])
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
        high_score = score
        nof = nof_list[n]
print("Optimum number of features: %d" %nof)
print("Score with %d features: %f" % (nof, high_score))
cols = list(X.columns)
model = LinearRegression()
rfe = RFE(model, step=2)             
X_rfe = rfe.fit_transform(X,y)  
model.fit(X_rfe,y)              
temp = pd.Series(rfe.support_,index = cols)
selected_features_rfe = temp[temp==True].index
print(selected_features_rfe)
reg = LassoCV()
reg.fit(X, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(X,y))
coef = pd.Series(reg.coef_, index = X.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (5.0, 5.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")
plt.show()
```
# OUPUT
DATA PREPROCESSING BEFORE FEATURE SELECTION:
![7 1](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/cbfcc142-038e-4bef-a9ae-70a5b9f51603)
![7 2](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/e8fff0b6-aeb6-45ad-9d12-627fa6021076)
![7 3](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/f4acc744-dd7f-48a6-91d5-963d44d220f4)
![7 4](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/4254941d-aec2-4dea-b9f1-1c784b85530c)
![7 5](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/a79e707f-80e1-493e-bae1-9da15cd89612)
![7 6](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/49a5cd00-8fd0-4824-b887-49aae33b095c)
![7 7](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/2e73067f-a0f8-4816-80a7-849fbd969a3a)
![7 8](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/d606b755-629e-47a9-aa3d-76754b590aea)
![7 9](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/4b66ca9a-865e-41d8-9201-9d75fc3569a5)
WRAPPER METHOD:
BACKWARD ELIMINATION:
![7 10](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/367bde4a-f647-444a-815b-85911d521f65)
![7 12](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/5329281e-462c-4f1c-86ce-b125cd696bd4)
FINAL SET OF FEATURE:
![7 13](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/bb48a7d0-328b-4091-9495-9c9293d153a6)
EMBEDDED METHOD:
![7 14](https://github.com/Kowsalyasathya/Ex-07-Feature-Selection/assets/118671457/d3b09c98-1bab-47ec-9dae-651b956f37cb)


## RESULT:

Thus, the various feature selection techniques have been performed on a given dataset successfully.
