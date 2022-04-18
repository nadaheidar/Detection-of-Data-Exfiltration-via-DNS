
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score , f1_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix,confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV


df = pd.read_csv("training_dataset.csv")

df.head(5)

train, test = train_test_split(df, test_size=0.3)

print(len(train))
print(len(test))

x_test=test.drop(['timestamp','longest_word','sld','Label'],axis=1)
y_test=test['Label']

x_train=train.drop(['timestamp','longest_word','sld','Label'],axis=1)
y_train=train['Label']

rfc = RandomForestClassifier( random_state=0)
# param_grid = {
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }
#
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# CV_rfc.fit(x_train, y_train)
# print(CV_rfc.best_params_)


rfc = rfc.fit(x_train , y_train)
rfc_y_pred= rfc.predict(x_test)

f_score = f1_score(y_true=y_test, y_pred=rfc_y_pred)*100
print("F1_score : ",f_score,"%")

#plot_confusion_matrix(y_test, rfc_y_pred)
cm = confusion_matrix(y_test, rfc_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("../reports/figures/confusion matrix randomforest.jpg")


