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

dataArray = np.array(X_train, dtype=int)
targetArray = dataArray[:, -1]
entropyLength = 0

def entropy(target_col):
    unique_classes, count_unique_classes = np.unique(target_col, return_counts=True)
    total = int(count_unique_classes[0]) + int(count_unique_classes[1])
    count_unique_class0 = int(count_unique_classes[0])
    count_unique_class1 = int(count_unique_classes[1])
    entropy = -(count_unique_class0/total*math.log(count_unique_class0/total, 2) + count_unique_class1/total*math.log(count_unique_class1/total, 2))
    return entropy


gainList = dict()
HD = entropy(targetArray)
numOfElements = len(X_train)
gain = 0
gain_array = []
Muli = 0
gainDictionary = {}

for name in dataNames:
    data = X_train.groupby(name)['target'].value_counts()/X_train.groupby(name)['target'].count()

    uniqueVAlues = np.unique(X_train[name])
    lengthOfAttributes = np.array(X_train.groupby(name)['target'].count())
    #for i in data[name]:
    #    print(i)
    # print(data[name].values)
    # print(data)
    # print(data[name].str[0])
    # print(data, "-------------------------")
    # print(name)
    # print(uniqueVAlues)
    entropyList = []

    #print(name)
    for i in uniqueVAlues:

    #    print("This is for attribute :", i)
        entropy = 0
        v = uV[0]

        p1 = data[i][v]
        p2 = 1 - p1

        entropy += p1 * math.log2(p1)
        if p2 != 0:
            entropy += p2 * math.log2(p2)

        entropy *= -1
        entropyList.append(entropy)

    # TODO: calculate gain
    gain = 0
    # for i in entropyList:





    for v in uV:
        Muli += data[i][v] * lengthOfAttributes[v]


    gain = HD - (1/numOfElements)*Muli
    Muli = 0
    gain_array.append(gain)
    gainDictionary[name] = gain
    gain = 0
print(gainDictionary)

