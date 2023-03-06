# import numpy as np
from sklearn.datasets import load_diabetes

#1. DATA
dataset = load_diabetes()

x = dataset.data
y = dataset.target

print(x.shape)
print(y.shape)

#2. MODEL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

model = Sequential()
model.add(Dense(516, input_dim = 10))
model.add(Dense(256))
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(1))

#3. COMPILE
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size= 0.7,
    random_state=24
)
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 3)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss:", loss)
from sklear.matrics import r2_score
y_predict = model.evaluate(x_test)
r2 = r2_score(y_test, y_predict)
print(r2)

