import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

# Stores the data as a dataframe
df = pd.read_csv("heart.csv")
# Split the data into test and training data
X_train, X_test, y_train, y_test = train_test_split(df, df["target"], test_size=0.3, random_state=21)

dataNames = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
entropy = dict()

uV = np.unique(X_train["target"])

for name in dataNames:
    data = X_train.groupby(name)['target'].value_counts() / X_train.groupby(name)['target'].count()

    uniqueVAlues, sumUnquie = np.unique(X_train[name], return_counts=True)

    # print(data, "-------------------------")
    # print(name)
    # print(uniqueVAlues)
    # for i in uniqueVAlues:
    #    entropy = 0
    #    for v in uV:
    #        print(data[i][v])

    # print(uniqueVAlues, sumUnquie)

# data = X_train.groupby("sex")['target'].value_counts()/X_train.groupby("sex")['target'].count()
# print(data, data[0][1])

