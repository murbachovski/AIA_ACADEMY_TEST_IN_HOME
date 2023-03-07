import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

#1. DATA
path = ('./_data/ddarung/')
path_save = ('./_save/ddarung')
#1-2. TRAIN
train_csv = pd.read_csv(path + 'train.csv')

#1-3. TEST
test_csv = pd.read_csv(path + 'test.csv')

#1-4. ISNULL(결측치 처리)
train_csv = train_csv.dropna()

#1-5. DROP(x, y DATA SPLIT)
x = train_csv(['count'], axis=1)
y = train_csv['count']

#1-6. TRAIN_TEST_SPLIT
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.7,
    random_state=32
)

#1-7. VALIDATION.TRAIN_TEST_SPLIT
x_train, x_val, y_train, y_val = train_test_split(
    x_train,
    y_train,
    train_size=0.3,
    random_state=16
)

#2. MODEL
model = Sequential()
model.add(Dense(256, input_dim=9))
model.add(Dense(128, activation='linear'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='relu'))

#3. COMPILE, VALIDATION(이곳에서도 가능)
model.compile(loss = 'mse', optimizer='adam')
model.fit(x_train, y_train, epochs = 10, batch_size=10, validation_data=(x_val, y_val))
#위에서 validation설정하면 여기서 어떻게 해주더라? 찾았다!!!
#3-1. COMPILE, VALIDATION(이곳에서 VALIDATION 해줄때)
# model.compile(loss = 'mse', optimizer='adam')
# model.fit(x_train, y_train, epochs = 10, batch_size=10, validation_split=0.3)


#4. EVALUATE, PREDICT
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

#5. DEF
def RMSE(y_test, y_predict):
    return np.sqrt(mean_absolute_error(y_test, y_predict))
rmse = RMSE(y_test, y_predict)

#6. SUBMISSION
y_submit = model.predict(test_csv)
submission = pd.read_csv(path + 'submission.csv')
submission['count'] = y_submit
submission.to_csv(path_save + '_new_submit.csv')