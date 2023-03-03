#1. DATA
#2. MODEL
#3. COMPILE
#4. EVALUATE, PREDICT

#1. DATA
# import numpy as np
# x = np.array([1, 2, 3])
# y = np.array([1, 2, 3])

#실제 데이터를 가져오려면?
from sklearn.datasets import load_diabetes
dataset = load_diabetes()
x = dataset.data
y = dataset.target

# print(x.shape)
# print(y.shape)

#2. MODEL
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(10, input_dim = 10))
model.add(Dense(10))    
model.add(Dense(10))    
model.add(Dense(10))    
model.add(Dense(11))    

#3. COMPILE
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,
    random_state=10
)
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 10)

#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
print("loss: ", loss)

from sklearn.metrics import r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2: ", r2)