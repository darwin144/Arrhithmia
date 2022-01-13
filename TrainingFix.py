from keras.models import Sequential
from keras.layers.core import Dense
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
import sklearn.metrics as met
import pandas as pd
import numpy as np


np.random.seed(7)
dataset = pd.read_csv('C:/Users/Darwin_Personal1/github/TA_Firdaus/Result/new_arrhytmiabaru99.csv', header =0)

X = dataset.drop(['QRS-msec','Q-T Interval'],axis=1)
y = dataset['Diagnosis']

X_train, X_test, y_train, y_test = ms.train_test_split(X,y, test_size =0.2)
#y_test.count()
#X_train.count()

model = Sequential()
model.add(Dense(200, input_dim=8, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = "binary_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(X_train,y_train, epochs=1000, batch_size=10, validation_split=0.015)

#training result
scores = model.evaluate(X_train,y_train)
print("\n%s : %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Testing
#print("Testing result :")
y_pred = model.predict(X_test)
#y_pred = y_pred > .5
print(met.classification_report(y_test, y_pred))
