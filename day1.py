#1. DATA
import numpy as np
# numpy = 수치화
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])


#2. MODEL
import tensorflow as tf 

from tensorflow.keras.models import Sequntial 
from tensorflow.keras.layers import Dense

model = Sequntial()
model.add(Dense(1, input_dim = 1))


#3. COMPILE
model.compile(losee = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 100)
#fit = 훈련을 시키다.
#epochs = 반복 횟수