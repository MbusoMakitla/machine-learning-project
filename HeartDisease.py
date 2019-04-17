import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import math

# Stores the data as a dataframe
df = pd.read_csv("heart.csv")
# Split the data into test and training data
X_train, X_test, y_train, y_test = train_test_split(df, df["target"], test_size=0.3, random_state=21)

# delete the target column
features = X_train.drop('target', 1)
# convert from dataframe to array
dataArray = np.array(X_train, dtype=int)
targetArray = dataArray[:, -1]
featuresArray = np.array(features, dtype=int)


# Calculate Entropy

def entropy(target_col):
    unique_classes, count_unique_classes = np.unique(target_col[:, -1], return_counts=True)
    total = int(count_unique_classes[0]) + int(count_unique_classes[1])
    count_unique_class0 = int(count_unique_classes[0])
    count_unique_class1 = int(count_unique_classes[1])
    entropy = -(count_unique_class0/total*math.log(count_unique_class0/total, 2) + count_unique_class1/total*math.log(count_unique_class1/total, 2))
    print("The entropy is: ", entropy)
    return


# We need to call entropy to verify if it is calculating the correct thing

