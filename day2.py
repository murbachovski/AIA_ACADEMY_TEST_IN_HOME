#코드 구성 순서 4단계
#1. DATA, 데이터
#2. MODEL, 모델 구성
#3. COMPILE, 훈련
#4. EVALUATE, 평가, 예측

#1. DATA
import numpy as np 
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

#2. MODEL
import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 

model = Sequential()
model.add(Dense(3, input_dim = 1))
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(1))

#3. COMPILE
model.compile(loss = 'mse', optimizer = 'adam')
#loss = MSA, MAE...등등 종류가 꽤 있다.
model.fit(x, y, epochs = 100, batch_size = 1)
#batch_size가 마냥 낮다고 좋은 것은 아니다. (속도 저하)
#default batch_size = 32

#4. 평가, 예측
loss = model.evaluate(x, y)
print("loss:", loss)

result = model.predict([4])
print("[4]의 예측 값은:", result)

# Y = W(weight)x + b

# [
#  [1, 2, 3],
#  [4, 5, 6]
#
#  ]
#각각은 스칼라 0차원
#벡터는 (3, ) 1차원
#행렬은 (2, 3) 2차원
#Tensor는 (4, 2, 3) 3차원
#4차원 Tensor -> 5차원 Tensor....->
#행무시, 열우선


#########################
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense  

#1. data
x = np.array(
    [[1, 1],
    [2, 1],
    [3, 1],
    [4, 1],
    [5, 2],
    [6, 1.3],  
    [7, 1.4],
    [8, 1.5],  
    [9, 1.6],
    [10, 1.4]] # 10행2열
)

y = np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

print(x.shape) # (10, 2) -> 2개의 특성을 가진 10개에 데이터
print(y.shape) # (10, )
# 열 = column 
# 행무시, 열우선
# 행렬 문제 10개 만들기. 목요일

# 2. model
model = Sequential()
model.add(Dense(3, input_dim = 2)) # Dense = 내 마음대로 // input_dim = 열, column
model.add(Dense(5)) 
model.add(Dense(4)) 
model.add(Dense(1))

# 3. Compile, 훈련
model.compile(loss = 'mae', optimizer = 'adam')
model.fit(x, y, epochs = 30, batch_size = 5)

#4. evaluate
loss = model.evaluate(x, y)
print("loss:", loss)

#5. result
result = model.predict([[10, 1.4]]) #열우선. 데이터 구조 맞추어 주기. 
print("[[10, 14]]의 예측값:", result)