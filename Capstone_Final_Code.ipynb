# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 15:16:45 2020

@author: carlosg1
"""

import numpy as np
import pandas as pd
import scipy
import seaborn as sns 
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
# model metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
#cross validation 
from sklearn.model_selection import train_test_split
#import models
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 
from sklearn.linear_model import SGDClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn. tree import DecisionTreeClassifier 
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

#Import Data
heart_data = pd.read_csv('heart-disease.csv')
#review data
heart_data.head()
heart_data.isnull().sum()
# code to convert numeric variables into categorical variables for analysis purpose
bins= [0,12,18,39,59,60]
labels = ['Child','Adolescence','Younger Adult','Adult','Senior Adult']
heart_data['AgeGroup'] = pd.cut(heart_data['age'], bins=bins, labels=labels, right=False)
heart_data['sex'] = heart_data.sex.replace([1,0], ['male', 'female'])
heart_data['chest pain'] = heart_data.cp.replace([0,1,2,3,4], ['no_cp','typical_ang', 'atypical_ang', 'non_anginal_pain', 'asymptomatic'])
heart_data['Fasting blood sugar'] = heart_data.fbs.replace([1,0], ['true', 'false'])
heart_data['Resting electrocardiographic'] = heart_data.restecg.replace([0,1,2], ['normal', 'st_abnormality', 'prob_lvh'])
heart_data['Exercise induced angina'] = heart_data.exang.replace([0,1], ['no', 'yes'])
heart_data['slope'] = heart_data.slope.replace([0,1,2,3], ['no_slope','upsloping', 'flat', 'downsloping'])
heart_data['Thalium heart scan'] = heart_data.thal.replace([3,6,7], ['normal', 'fixed_def', 'rev_def'])
heart_data['Diagnosis of heart disease'] = heart_data.target.replace([1,0], ['yes', 'no'])
heart_data.head()
#g = sns.pairplot(heart_data, vars =['age', 'trestbps', 'chol', 'thalach', 'oldpeak' ], hue = 'target')
#g.map_diag(sns.distplot)
#g.add_legend()
#g.fig.suptitle('FacetGrid plot', fontsize = 20)
#g.fig.subplots_adjust(top= 0.9);
#Another code to stilying
#corr = heart_data.corr()
#print(corr)
#corr.style.background_gradient(cmap='coolwarm')
#corr.style.background_gradient(cmap='BrBG')

# Importing dataset for corr matrix
heart_data_corr = pd.read_csv('heart-disease.csv')
heart_data_corr.head()
heart_data_corr.isnull().sum()

# Plotting correlation matrix
sns.set(rc={'figure.figsize':(26,10)})
sns.set_context("talk", font_scale=0.7)
sns.heatmap(heart_data_corr.corr(), cmap='Reds', annot=True);


#Firstly, let's look at the distribution.
plt.figure(figsize=(10,4))
plt.legend(loc='upper left')
g = sns.countplot(data = heart_data, x = 'age', hue = 'target')
g.legend(title = 'Heart disease patient?', loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1)
#import seaborn as sns
#rc={'axes.labelsize': 40, 'font.size': 40, 'legend.fontsize': 40.0, 'axes.titlesize': 40}
#sns.set(rc=rc)
#
#plot = sns.barplot(data = age_corr_y, x = 'AgeGroup', y = 'count')
#plot.set_ylabel('count', fontsize=30)
#plot.set_xlabel('AgeGroup', fontsize=30)
#plot.set_title('Correlation graph for Age vs heart disease patient', fontsize = 30)
#correlation by age group
age_corr = ['AgeGroup', 'Diagnosis of heart disease']
age_corr1 = heart_data[age_corr]
age_corr_y = age_corr1[age_corr1['Diagnosis of heart disease'] == 'yes'].groupby(['AgeGroup']).size().reset_index(name = 'count')
age_corr_y.corr()
sns.barplot(data = age_corr_y, x = 'AgeGroup', y = 'count').set_title("Correlation graph for Age vs heart disease patient", fontsize=25)

age_corr_n = age_corr1[age_corr1['Diagnosis of heart disease'] == 'no'].groupby(['AgeGroup']).size().reset_index(name = 'count')
age_corr_n.corr()
sns.lineplot(data = age_corr_n, x = 'AgeGroup', y = 'count').set_title("Correlation graph for Age vs healthy patient", fontsize=25)


# Showing number of heart disease patients based on sex
sex_corr = ['sex', 'Diagnosis of heart disease']
sex_corr1 = heart_data[sex_corr]
sex_corr_y = sex_corr1[sex_corr1['Diagnosis of heart disease'] == 'yes'].groupby(['sex']).size().reset_index(name = 'count')
sex_corr_y
# Showing number of without heart disease patients based on sex
sex_corr_n = sex_corr1[sex_corr1['Diagnosis of heart disease'] == 'no'].groupby(['sex']).size().reset_index(name = 'count')
sex_corr_n

# Plot of Heart disease patient based on sex 
g1 = sns.boxplot(data = heart_data, x = 'sex', y = 'age', hue = 'Diagnosis of heart disease',palette="Set3")
g1.legend(title = 'Heart disease patient', loc='center left', bbox_to_anchor=(1, 0,5), ncol=1, fontsize = 25)
g1.set_title('Boxplot showing age vs sex', fontsize = 25)

# Showing number of heart disease patients based on chest pain
cp_corr = ['chest pain', 'Diagnosis of heart disease']
cp_corr1 = heart_data[cp_corr]
cp_corr_y = cp_corr1[cp_corr1['Diagnosis of heart disease'] == 'yes'].groupby(['chest pain']).size().reset_index(name = 'count')
cp_corr_y.corr()
sns.barplot(data = cp_corr_y, x = 'chest pain', y = 'count').set_title("Correlation graph for chest pain vs heart disease patient",fontsize = 25)

# Showing number of healthy patients based on cp 
cp_corr_n = cp_corr1[cp_corr1['Diagnosis of heart disease'] == 'no'].groupby(['chest pain']).size().reset_index(name = 'count')
cp_corr_n.corr()
sns.barplot(data = cp_corr_n, x = 'chest pain', y = 'count').set_title("Correlation graph for chest pain vs healthy patient",fontsize=25)

# Showing number of heart disease patients based on trestbps
restbp_corr = ['trestbps', 'Diagnosis of heart disease']
restbp_corr1 = heart_data[restbp_corr]
restbp_corr_y = restbp_corr1[restbp_corr1['Diagnosis of heart disease'] == 'yes'].groupby(['trestbps']).size().reset_index(name = 'count')
restbp_corr_y.corr()
sns.lineplot(data = restbp_corr_y, x = 'trestbps', y = 'count').set_title('Correlation between resting blood pressure and healthy patients',fontsize=25)

# Showing number of heart disease patients based on serum cholesterol
chol_corr = ['chol', 'Diagnosis of heart disease']
chol_corr1 = heart_data[chol_corr]
chol_corr1.chol = chol_corr1.chol.round(decimals=-1)
chol_corr_y = chol_corr1[chol_corr1['Diagnosis of heart disease'] == 'yes'].groupby(['chol']).size().reset_index(name = 'count')
chol_corr_y.corr()
sns.scatterplot(data = chol_corr_y, x = 'chol', y = 'count').set_title('Correlation between serum cholesterol and heart disease patients',fontsize=25)

# Showing number of without heart disease patients based on serum cholesterol
chol_corr_n = chol_corr1[chol_corr1['Diagnosis of heart disease'] == 'no'].groupby(['chol']).size().reset_index(name = 'count')
chol_corr_n.corr()
sns.scatterplot(data = chol_corr_n, x = 'chol', y = 'count').set_title('Correlation between serum cholesterol and healthy patients')

# Showing number of heart disease patients based on fasting blood sugar
fbs_corr = ['Fasting blood sugar', 'Diagnosis of heart disease']
fbs_corr1 = heart_data[fbs_corr]
fbs_corr_y = fbs_corr1[fbs_corr1['Diagnosis of heart disease'] == 'yes'].groupby(['Fasting blood sugar']).size().reset_index(name = 'count')
fbs_corr_y
sns.barplot(data = fbs_corr_y, x = 'Fasting blood sugar', y = 'count').set_title('Correlation between Fasting blood sugar and heart disease patients',fontsize=25)

# Showing number of healthy patients based on fasting blood sugar
fbs_corr_n = fbs_corr1[fbs_corr1['Diagnosis of heart disease'] == 'no'].groupby(['Fasting blood sugar']).size().reset_index(name = 'count')
fbs_corr_n.corr()
sns.barplot(data = fbs_corr_n, x = 'Fasting blood sugar', y = 'count').set_title('Correlation between serum cholesterol and healthy patients')

# Showing number of heart disease patients based on resting ECG results
restecg_corr = ['Resting electrocardiographic', 'Diagnosis of heart disease']
restecg_corr1 = heart_data[restecg_corr]
restecg_corr_y = restecg_corr1[restecg_corr1['Diagnosis of heart disease'] == 'yes'].groupby(['Resting electrocardiographic']).size().reset_index(name = 'count')
restecg_corr_y
sns.barplot(data = restecg_corr_y, x = 'Resting electrocardiographic', y = 'count').set_title('Correlation between serum cholesterol during Resting electrocardiographic and heart disease patients',fontsize=25)

# Showing number of without heart disease patients based on resting ECG results
restecg_corr_n = restecg_corr1[restecg_corr1['Diagnosis of heart disease'] == 'no'].groupby(['Resting electrocardiographic']).size().reset_index(name = 'count')
restecg_corr_n
sns.barplot(data = restecg_corr_n, x = 'Resting electrocardiographic', y = 'count').set_title('Correlation between serum cholesterol during Resting electrocardiographic and healthy patients')

# Showing number of heart disease patients based on maximum heart rate
heartrate_corr = ['thalach', 'Diagnosis of heart disease']
heartrate_corr1 = heart_data[heartrate_corr]
heartrate_corr_y = heartrate_corr1[heartrate_corr1['Diagnosis of heart disease'] == 'yes'].groupby(['thalach']).size().reset_index(name = 'count')
heartrate_corr_y.corr()
sns.lmplot(data = heartrate_corr_y, x = 'thalach', y = 'count')

# Showing number of healty patients based on maximum heart rate
heartrate_corr_n = heartrate_corr1[heartrate_corr1['Diagnosis of heart disease'] == 'no'].groupby(['thalach']).size().reset_index(name = 'count')
heartrate_corr_n.corr()
sns.lmplot(data = heartrate_corr_n, x = 'thalach', y = 'count')

#Prepare data for modeling
#let us get rid of multicollinearity with target variable with PCA
#Import PCA Dataframe
heart_data_pred = pd.read_csv('heart-disease.csv')
scaler = StandardScaler()
scaler.fit(heart_data_pred)
scaled_data = scaler.transform(heart_data_pred)
pca = PCA(n_components=2)
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
scaled_data.shape
x_pca.shape

#Plot PCA information
plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=heart_data_pred['target'],cmap='rainbow')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
pca.components_

#start modeling
#Get models parameters
#Features
features = heart_data_pred.iloc[:,1:13]
#dependent variable
depVar = heart_data_pred['target']
#Separate training/test data
x_train, x_test, y_train, y_test = train_test_split(features, depVar, test_size=0.2)

#Models and tunning
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier(n_neighbors=15)))
models.append(('SVM', SVC(kernel='rbf', C=0.025,random_state=101)))
models.append(('SVR', SVC(kernel='rbf', C=10, gamma='auto')))
models.append(('DTree', DecisionTreeClassifier(max_depth=10,random_state=101,max_features=None, min_samples_leaf=15)))
models.append(('SGD', SGDClassifier(loss='modified_huber',shuffle=True,random_state=101))) 
models.append(('NB', GaussianNB()))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=101)
    cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Predict with LogisticRegression
lr = LogisticRegression() 
lr.fit(x_train,y_train) 
y_predict = lr.predict (x_test)
predRsquaredlr = r2_score(y_test,y_predict)
rmselr = sqrt(mean_squared_error(y_test, y_predict))
print('R Squared: %.3f' % predRsquaredlr)
print('RMSE: %.3f' % rmselr)

#Naïve Bayes
#from sklearn.naive_bayes import GaussianNB 
#nb = GaussianNB() 
#nb.fit(x_train,y_train) 
#y_predict= nb.precit(x_test) 

#SVM
#from sklearn.svm import SVC 
#SVM = SVC(kernel='rbf', C=0.025,random_state=101)
#SVM.fit(X_train,y_train)
#y_pred_svc = SVM.predict(X_test)

#XGboost 
#from xgboost import XGBClassifier
#xgb_classifier = XGBClassifier()
#xgb_classifier.fit(X_train,y_train)
#y_pred_xgb = xgb_classifier.predict(X_test)

#Stochastic Gradient Descent
#from sklearn.linear_model import SGDClassifier 
#sgd = SGDClassifier(loss='modified_huber',shuffle=True,random_state=101) 
#sgd.fit(X_train,y_train) 
#y_predict = sgd.predict (x_test)

#KNN
#from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=15)
#knn.fit(X_train,y_train)
#y_predict=knn.predict(x_test)
 
#Decision Tree
#from sklearn. tree import DecisionTreeClassifier 
#dtree = DecisionTreeClassifier(max_depth=10,random_state=101,max_features=None, min_samples_leaf=15)
#dtree.fit(X_train,y_train) 
#y_predict = dtree.predict (x_test) 

#SVR
#modelSVR = SVR()
#from sklearn.svm import SVR
#SVR= SVR(kernel='rbf', C=10, gamma='auto')
#SVR.fit(X_train,y_train)
#y_predict = SVR.predict (x_test)

#Random Forest
#from sklearn.ensemble import RandomForestRegressor
#RF = RandomForestRegressor(n_estimators=40, verbose=2,random_state=5,warm_start=True,oob_score=True)
#RF.fit(X_train,y_train)
#print(cross_val_score(RF, x_train, y_train)) 
#y_predict = RF.predict (x_test)

#Linear Regression
#from sklearn.linear_model import LinearRegression
#LR = LinearRegression(copy_X= True, fit_intercept= True,normalize= True)
#LR.fit(X_train,y_train)
#y_predict = LR.predict (x_test)






