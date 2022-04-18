#imports
import pandas as pd
import pickle
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, f1_score

def predict_model():
    df_features = pd.read_csv("../data/features.csv")
    df_features_all = pd.read_csv("../data/all_features.csv")
    path_saved_model = '../models/saved model.sav'
    with open(path_saved_model, 'rb') as pickle_file:
        rfc = pickle.load(pickle_file)

    #predict label from data also predict score

    predicted_label = rfc.predict(df_features)
    score = np.max(rfc.predict_proba(df_features),axis=1)

    df_features_all['predicted_label'] = predicted_label
    df_features_all['score'] = score

    # print(df_all_features.head())
    df_features_all.to_csv("../data/predicted.csv",index=False)

