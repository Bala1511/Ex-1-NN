<H3>ENTER YOUR NAME: BALA MURUGAN </H3>
<H3>ENTER YOUR REGISTER NO:212222230017</H3>
<H3>EX. NO.1</H3>
<H3>DATE29.4.24</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
#IMPORT THE REQUIRED LIBRARIES
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#READ THE DATASET
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

#REMOVE THE UNWANTED COLUMNS USING DROP OPERATION
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#CHECK FOR THE NULL VALUES
df.isnull().sum()

#CHECK FOR DUPLICATED VALUES
df.duplicated()

#DESCRIBE THE DATASET
df.describe()

#PREPROCESSING THE DATASET
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#SPLIT THE DATASET INTO TRAINING AND TESTING DATA
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```

## OUTPUT:
SHOW YOUR OUTPUT HERE
### DATASET
![image](https://github.com/Bala1511/Ex-1-NN/assets/118680410/02e175aa-473e-4549-a6b9-d5a26a481e7a)

### AFTER DROP OPERATION
![image](https://github.com/Bala1511/Ex-1-NN/assets/118680410/19eaf5f5-30df-4809-a341-9856d278ed3f)
### CHECK NULL VALUES
![image](https://github.com/Bala1511/Ex-1-NN/assets/118680410/dd0be44c-ca22-49f8-91ad-f29c3d92263c)
### DESCRIBE DATA
![image](https://github.com/Bala1511/Ex-1-NN/assets/118680410/dde46432-db17-40c5-9abe-ef982d014c81)


![image](https://github.com/Bala1511/Ex-1-NN/assets/118680410/328b3c34-b2c5-4d06-974f-a8b177edc969)


![image](https://github.com/Bala1511/Ex-1-NN/assets/118680410/96b56ecf-2a38-4547-a2bc-e5a3dc5efa7b)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


