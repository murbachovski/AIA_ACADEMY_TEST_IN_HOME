#1. DATA
#2. MODEL
#3. COMPILE
#4. EVALUATE, PREDICT

#1. DATA
import numpy as np
x = np.array([1, 2, 3, 4])    
y = np.array([1, 2, 3, 4])

#2. MODEL
from tensorflow.keras.models import Sequential  
from tensorflow.keras.layers import Dense     
model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(4))  
model.add(Dense(5))  
model.add(Dense(1))  

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x, y, epochs = 100, batch_size = 3)

#4. EVALUATE, PREDICT
loss = model.evaluate(x, y)
print("loss:", loss)
result = model.predict([5])
print("result:", result)
####################################################################1. 가장 기본적인 코드.#################


#1. DATA
import numpy as np
x = np.array([
    [1, 2, 3],
    [4, 5, 6]
])
y = np.array([
    [1, 2, 3],
    [4, 5, 6]
])     

#2. MODEL
import tensorflow as tf      
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Dense
model = Sequential()
model.add(Dense(3, input_dim = 3))   # 갑자기 input_dim이 햇갈리네...? 행을 받나? 열을 받나? / 행을 받는거 같아! / 땡 열, Column이랍닏나
model.add(Dense(4))     
model.add(Dense(5))        
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs = 100, batch_size = 3)

#4. EVALUATE, PREDICT
loss = model.evaluate(x, y)
print('loss:', loss)
result = model.predict([7]) # 여기에는 무엇을 예측해야하지?
####################################################################2. 데이터 구조가 단순하지 않은 경우#################

#1. DATA
import numpy as np 
x = np.array([
    [1, 2],
    [3, 4],
    [5, 6],
    [7, 8],
    [9, 10],
    [11, 12],
    [13, 14],
    [15, 16],
    [17, 18],
    [19, 20],
])

y = np.array([
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20
])

#2. MODEL
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(3, input_dim = 2))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x, y, epochs = 100, batch_size = 3)

#4. evaluate
loss = model.evaluate(x, y)
print("loss:", loss)

#5. result
result = model.predict([[10, 1.4]])
print("[[10, 14]]의 예측값:", result)
####################################################################3. 데이터 구조가 단순하지 않은 경우2#################


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense   

#1. DATA
x = np.array([range(10), range(21, 31), range(201, 211)])

print(x)
x = x.T 
print(x.shape)  

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
y = y.T

# 2. model
model = Sequential()
model.add(Dense(3, input_dim = 3))
model.add(Dense(5)) 
model.add(Dense(4)) 
model.add(Dense(1)) 

# 3. Compile
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x, y, epochs = 30, batch_size = 3)

#4. evaluate, result
loss = model.evaluate(x, y)
print("loss:", loss)
result = model.predict([[9, 30, 210]])
print("[[9, 30, 210]]의 예측값:", result)
###########################################################4. 데이터를 transpose, 전치가 필요한 경우(x, y둘다 전치 가능)###############


import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

#1. DATA
x = np.array([range(10), range(21, 31), range(201, 211)])
print(x.shape)  
new_x = x.T 

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]  
              ])
new_y = y.T          

#2. MODEL
model = Sequential()
model.add(Dense(2, input_dim = 3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(new_x, new_y, epochs = 100)

#4. 평가, 예측
loss = model.evaluate(new_x, new_y)
print("loss: ", loss)
#5. RESULT
result = model.predict([[9, 30, 210]])
print("result[[9, 30, 210]]의 예측값:", result)
####################################################################5. x_data = N, y_data = N개인 경우#################


import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. DATA
x = np.array([range(5), range(21, 26), range(101, 106)])
print(x)
print(x.shape)
new_x = x.T
print(new_x.shape)
y = np.array([
    [1, 2, 3, 4, 5],
    [1.1, 1.2, 1.3, 1.4, 1.5],
    [10, 20, 30, 40, 50]
])
new_y = y.T
print(new_y.shape)
#2. MODEL
model = Sequential()
model.add(Dense(2, input_dim = 3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(3)) #outputData구조 잘 맞추기. 행무시/열우선
#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(new_x, new_y, epochs = 10)
#4. EVAlUATE
loss = model.evaluate(new_x, new_y)
print("loss: ", loss)
#5. RESULT
result = model.predict([[4, 25, 105]])
print("result[4, 25, 105]", result)

#########new_x and new_y 데이터 구조 잘 맞추어 주기.#########
#########################################################6. x_data = N, y_data = N개인 경우(x_data를 range로 표시)#################


import numpy as np
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

#1. DATA
x = np.array([range(10)])
print(x.shape)    # 1, 10
new_x = x.T       # 10, 1

y = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
              [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
              ])
new_y = y.T       # (10, 3)

# [실습]
# [예측]: [[9]] ==> y값 [[10, 1.9, 0]]

#2. MODEL
model = Sequential() # InputData
model.add(Dense(3, input_dim = 1)) #First Hidden Layout
model.add(Dense(3))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3)) #OutputData

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(new_x, new_y, epochs = 50)

#4. 평가, 예측
loss = model.evaluate(new_x, new_y)
print("loss: ", loss)

#5. RESULT
result = model.predict([[9]]) # 보고 싶은 x데이터 넣어주기.
print("result[[9]]:", result)
#########################################################7. x_data = N, y_data = N개인 경우################

import numpy as np  
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  

#1. DATA
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
# 마지막에 온 ,는 딱히 상관없다.
# print(x)
# print(y)

x_train = np.array([1, 2, 3, 4, 5, 6, 7])
y_train = np.array([1, 2, 3, 4, 5, 6, 7])
x_test = np.array([8, 9, 10])
y_test = np.array([8, 9, 10])

#2. MODEL
model = Sequential()
model.add(Dense(10, input_dim = 1)) 
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x_train, y_train, epochs = 100, batch_size = 2)

#4. EVALUATE
loss = model.evaluate(x_test, y_test)
print("loss:", loss)

#5. RESULT
result = model.predict([11])
print("[11]predict:", result)
#########################################################8. train, test Data 나눠주기 기초1################


import numpy as np  
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense  

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    # train_size=0.7,
    test_size=0.3,
    random_state=1234,
    # shuffle=False,
)
print(x_train)
print(x_test)

#2. MODEL
model = Sequential()
model.add(Dense(1, input_dim = 1))

#3. COMPILE
model.compile(loss = 'mse', optimizer = 'adam') 
model.fit(x_train, y_train, epochs = 100, batch_size = 1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
result = model.predict([11])
print('[11]predict:', result)
#########################################################9. train, test Data /Shuffle로 구분하기!!################
