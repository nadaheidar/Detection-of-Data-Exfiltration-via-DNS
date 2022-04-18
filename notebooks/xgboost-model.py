
import pandas as pd
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , plot_confusion_matrix, f1_score
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix,confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv("training_dataset.csv")

df.head(5)

class_0 = len(df[df['Label'] == 0])
class_1 = len(df[df['Label'] == 1])


height=[class_0,class_1]
bars = ('class_0','class_1')
x_pos = np.arange(len(bars))
plt.bar(x_pos, height, color=['yellow', 'cyan'])
plt.xticks(x_pos, bars)
plt.savefig("../reports/figures/barchart for all data XGBOOST.jpg")

train, test = train_test_split(df, test_size=0.3)

x_test=test.drop(['timestamp','longest_word','sld','Label'],axis=1)
y_test=test['Label']

x_train=train.drop(['timestamp','longest_word','sld','Label'],axis=1)
y_train=train['Label']

xg = GradientBoostingClassifier()

xg = xg.fit(x_train , y_train)
xg_y_pred= xg.predict(x_test)
xg_y_pred

f_score = f1_score(y_true=y_test, y_pred=xg_y_pred)*100
print("F1_score_xgboost : ",f_score,"%")


cm = confusion_matrix(y_test, xg_y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig("../reports/figures/confusion matrix XGBOOST.jpg")

class_0_test=0
class1_test=0
for i in y_test:
  if i == 0:
    class_0_test +=1
  else:
    class1_test +=1

height=[class_0_test,class1_test]
bars = ('test class = 0','test class = 1')
x_pos = np.arange(len(bars))
plt.bar(x_pos, height, color=['yellow', 'cyan'])
plt.xticks(x_pos, bars)
plt.show()

class_0_pred=0
class1_pred=0
for i in xg_y_pred:
  if i == 0:
    class_0_pred +=1
  else:
    class1_pred +=1

height=[class_0_pred,class1_pred]
bars = ('predicted class = 0','predicted class = 1')
x_pos = np.arange(len(bars))
plt.bar(x_pos, height, color=['yellow', 'cyan'])
plt.xticks(x_pos, bars)
plt.savefig("../reports/figures/barchart XGBOOST.jpg")
