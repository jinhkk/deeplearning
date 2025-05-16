import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random

import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# 1. 데이터 생성
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


random.seed(777)


def get_data() :
    grad_a = 2.3
    intercept_b = 0.88

    error = 0.1

    num_data_points = 20

    x_values = []
    y_values = []

    for _ in range(num_data_points):
        x = random.uniform(1, 20)
        linear_val = grad_a * x + intercept_b
        noise = random.uniform(1 - error, 1 + error)
        y_with_noise = linear_val * noise
        x_values.append(x)
        y_values.append(y_with_noise)

    train_x = torch.from_numpy(np.array(x_values)).view(num_data_points, 1)
    train_y = torch.from_numpy(np.array(y_values))

    return train_x, train_y


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD # 학습률 지정하기 위한 옵티마이저 임포트

x, y = get_data()
model = Sequential()

model.add(Dense(1, input_dim=1, activation='linear'))

model.compile(optimizer=SGD(learning_rate=1e-6), loss='mse')
model.fit(x.numpy(), y.numpy(), epochs=20000) # TensorFlow 모델은 TensorFlow 텐서 또는 NumPy 배열이 필요하다

plt.scatter(x, y)
plt.plot(x, model.predict(x),'r')
plt.show()

