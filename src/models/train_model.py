#imports
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, f1_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

#read Csv file as dataframe
df = pd.read_csv("../data/training_dataset.csv")

#split data into train 70% and test 30%
train, test = train_test_split(df, test_size=0.3)

#drop features from dataset  and split data into features and label
x_test = test.drop(['timestamp', 'longest_word', 'sld', 'Label'], axis=1)
y_test = test['Label']

x_train = train.drop(['timestamp', 'longest_word', 'sld', 'Label'], axis=1)
y_train = train['Label']

#apply random forest classifier
rfc = RandomForestClassifier(random_state=0)
rfc = rfc.fit(x_train, y_train)
rfc_y_pred = rfc.predict(x_test)

#calc F1-score
f_score = f1_score(y_true=y_test, y_pred=rfc_y_pred) * 100
print("F1_score : ", f_score, "%")

# #confusion matrix on test data
# plot_confusion_matrix(rfc, x_test, y_test)
# plt.savefig("../visualization/cf.jpg")
# print("confusion matrix", plt.show())

# Save random forest model Using Pickle
import pickle
# save the model
filename = './saved model.sav'
pickle.dump(rfc, open(filename, 'wb'))
# load the model
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(x_test, y_test)
print(result)

