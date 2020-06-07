# Support Vector Regression (SVR)
# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def main():
    # Importing the dataset
    dataset = pd.read_csv('Car_Purchasing_Data.csv',encoding='Latin-1')
    x = dataset.iloc[:, 3:-1].values
    y = dataset.iloc[:, -1:].values
    x_or = x
    y_or = y
    #y = y.reshape(len(y),1)
    
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    y_train = sc_y.fit_transform(y_train)
    x_test = sc_x.transform(x_test)
    # Training the SVR model on the whole dataset
    from sklearn.svm import SVR
    regressor = SVR(kernel = 'rbf')
    regressor.fit(x_train, y_train)

    # Predicting a new result
    y_pred = sc_y.inverse_transform(regressor.predict(x_test))
    result = y_test
    #print(len(y_test))
    with open('pred.csv','w') as f:
        f.writelines(str(y_pred))
    print(y_test)
   
if __name__ == "__main__":
    main()