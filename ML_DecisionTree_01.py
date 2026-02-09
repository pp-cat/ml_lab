import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import random


# import dataset and do data preparation 
adult_tmp = pd.read_csv("C:\\Users\\pixal\\Desktop\\WPy64-312101\\notebooks\\adult\\adult_with_heading.csv")
adult_tmp['cap-gain-loss'] = adult_tmp['capital-gain'] + adult_tmp ['capital-loss']

adult_tmp.loc[:,'marital-status'] = adult_tmp['marital-status'].str.lstrip()  # column has prefix space

mar_cat = adult_tmp[['marital-status']]
mar_cat = mar_cat.replace({'Married-AF-spouse':'Married', 'Married-civ-spouse':'Married', 'Married-spouse-absent':'Married'})
dummies = pd.get_dummies(mar_cat['marital-status'], dtype='uint8')

adult_tmp = pd.concat((adult_tmp, dummies), axis=1)


# partition dataset
adult_train, adult_test = train_test_split(adult_tmp, test_size= 0.25, random_state = 7)

# reset index as the index is random in partition !!
adult_train = adult_train.reset_index()
adult_test = adult_test.reset_index()

# extract data set for training
y = adult_train[['class']]
x = adult_train[['cap-gain-loss', 'Divorced', 'Married', 'Never-married',
       'Separated', 'Widowed']]

# build Decision Tree
cart01 = DecisionTreeClassifier(criterion = "gini", max_leaf_nodes=5).fit(x,y)


# make columns heading for Decision Tree graph
y_names=["<=50K", ">50K"]
x_names=["cap-gain-loss", "Divorced", "Married", "Never-married", "Separated",	"Widowed"]

# export Decision Tree Graph
export_graphviz(cart01, out_file="D:\\wb\\python\\ML\\adult\\cart01.dot", 
feature_names=x_names, class_names=y_names)

# test model
x_test = adult_test[['cap-gain-loss', 'Divorced', 'Married', 'Never-married',
       'Separated', 'Widowed']]

predClassCART = cart01.predict(x_test)

predClassCART = pd.Series(predClassCART, name='predict') # convert array to Series
adult_predict = pd.concat((adult_test, predClassCART), axis=1) 
