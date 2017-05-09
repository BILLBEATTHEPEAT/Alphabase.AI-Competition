#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler


def main():
    ##################################
    # Step 1. Load alphabase.ai data
    print("Loading the data ...")
    # Load the data from the CSV files
    train = pd.read_csv('train.csv', header=0)
    test = pd.read_csv('test.csv', header=0)

    ##
    #################################
    ##################################
    # Step 2. Train the Logistic Regression Model
    # Prepare data, ignoring the NA-flag features
    Y = train['Target']
    X_train = np.array(train)[:, :55]
    ID = test['ID']
    X_test = np.array(test)[:, 1:56]

    # Missing values imputation
    print("Missing values imputation ...")
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X_train)
    X_train_alpha = imp.transform(X_train)
    X_test_alpha = imp.transform(X_test)

    scaler = MinMaxScaler()
    scaler.fit(X_train_alpha)
    X_train = (scaler.transform(X_train_alpha)).tolist()
    X_test = (scaler.transform(X_test_alpha)).tolist()

    # Create a Logistic Regression Model.
    print("Building document")
    np.save('X_train', X_train)
    del X_train
    np.save('X_test', X_test)
    del X_test
    np.save('Label', Y)
    del Y
    np.save('ID', ID)
    del ID


# Here the main program.


if __name__ == '__main__':
    main()
