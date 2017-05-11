#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


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
    Y = Y.tolist()
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

    model = SelectKBest(chi2, k=2)
    model.fit(X_train, Y)
    print model.get_support()

    X_train = model.transform(X_train)
    X_test = model.transform(X_test)
    print len(X_train[0])
    print X_train[0]

    print("Building document")
    np.save('X_train_1', X_train)
    del X_train
    np.save('X_test_1', X_test)
    del X_test
    # np.save('Label_1', Y)
    # del Y
    # np.save('ID', ID)
    # del ID


# Here the main program.


if __name__ == '__main__':
    main()
