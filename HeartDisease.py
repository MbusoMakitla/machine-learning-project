import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

# Stores the data as a dataframe
df = pd.read_csv("heart.csv")
# Split the data into test and training data
X_train, X_test, y_train, y_test = train_test_split(df, df["target"], test_size=0.3, random_state=21)

dataNames = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca"]
entropy = dict()

uV = np.unique(X_train["target"])

for name in dataNames:
    data = X_train.groupby(name)['target'].value_counts()/X_train.groupby(name)['target'].count()

    uniqueVAlues, sumUnquie = np.unique(X_train[name], return_counts=True)

    # print(data, "-------------------------")
    print(name)
    # print(uniqueVAlues)
    for i in uniqueVAlues:
        print("This is for attribute :", i)
        entropy = 0
        for v in uV:
            print(data[i][1])


#print(len(X_train))
# data = X_train.groupby("sex")['target'].value_counts()/X_train.groupby("sex")['target'].count()
# print(data, data[0][1])

#My analogy was that for each attrib under sex e.g 0.&1 You calculate the entropies for both, then store the unique value and the attrib
#e.g [[sex,1/3,2/3],[cp,0.75,0.5]...]We will then calculate the Gain(D,sex) for sex knowing those two entropies
#multiplied by the number of times their attributes 0/1 appears
# a third suggestion would be to store the number of occurences in the same array e.g [[sex,1/3,4,2/3,5]...]
#1/3 appears 4 times and 2/3 appears 5 times