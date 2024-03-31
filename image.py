#importing dependencies
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
matplotlib inline

#using pandas to read the database
data = pd.read_csv('mnist.csv')

#viewing column head
data.head()

#extracting data from the dataset
a = data.iloc[3,1:].values

#reshaping the extracted data into a reasonable size
a = a.reshape(28,28).astype('uint8')
plt.imshow(a)

#preparing the data
#seperating labels and data values
df_x = data.iloc[:,1:]
df_y = data.iloc[:,0]

#creating test and train sizes/batches
x_train, x_test, y_train, y_test = train_test_split

#check data
y_train.head()

#call rf classifier
rf = RandomForestClassifier(n_estimators=100)

#fit the model
rf.fit(x_train, y_train)

#prediction on test data
pred = rf.predict(x_test)
pred

#check prediction accuracy
s = y_test.values

#calculate number of correctly predicted values
count = 0
for i in range(len(pred)):
    if pred[i] == s[i]
    count = count+1
count

#total values that the prediction code was run on
len(pred)

#accuracy value
8090/8400

