import torch
import random
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 1. 데이터를 생성

"""
    데이터 생성 함수
    함수명 : get_data()
    입력 파라미터 : None
    출력 파라미터 : x, y data
    설명 : 선형 방정식 y = grad * x + intercept 를 바탕으로 
    랜덤한 오차를 포함한 x, y 데이터 생성
"""

random.seed(777)


def get_data():
    """
    y = 2.3x + 0.88
    x 값을 1 ~ 20 랜덤 추출하고 출력된 y 값에 (+-10%)범위의 랜덤 오차를 곱하여 생성
    :return:
    """
    # 선형 방정식 계수
    grad_a = 2.3
    intercept_b = 0.88

    # y 오차 범위 (+-10)
    error = 0.1

    # number of data
    num_data_points = 20

    # x, y 리스트 초기화
    x_values = []
    y_values = []

    # x, y 데이터 생성
    for _ in range(num_data_points):
        # 1 ~ 20 범위에서 x를 랜덤으로 추출
        x = random.uniform(1, 20)
        # 선형 방정식 + 랜덤 오차 포함 y 계산
        linear_val = grad_a * x + intercept_b
        noise_factor = random.uniform(1 - error, 1 + error)
        y_with_factor = linear_val * noise_factor
        x_values.append(x)
        y_values.append(y_with_factor)

    # torch.tensor 로 반환
    train_x = torch.from_numpy(np.array(x_values)).view(num_data_points, 1)
    train_y = torch.from_numpy(np.array(y_values))

    return train_x, train_y


"""
func name : get_weights()
--description
"""


def get_weights():
    """
    weight(gradient), bias(intercept) : 표준 정규 분포에서 랜덤 초기화

    :return:
    """
    w = torch.randn(1) # randn --> return float
    w.requires_grad = True
    b = torch.randn(1)
    w.requires_grad = True

    return w, b

"""
가설함수 정의
function name : simple_network()
- input, output parameter, etc. 정리해볼 것
"""

def simple_network(x, w, b):
    """
    입력 x: N*1 matrix ==> 20x1
    :return:
    """
    y_pred = torch.matmul(x, w.double()) + b

    return y_pred

"""
손실함수 정의
func : loss_func(y, y_pred)
define : in, out
function descript
"""
def loss_func(y, y_pred):
    loss = torch.mean((y - y_pred).pow(2).sum())

    for param in [w, b]:
        if param.grad is not None :
            param.grad.data.zero_()


    loss.backward()

    return loss.data

"""
update param
"""

def update_param(lr):
    w.data = w.data - lr * w.grad.data
    b.data = lr * w.grad.data

import matplotlib.pyplot as plt
## plot
def plot_variable(x, y, z='', **kwargs):
    x_data = x.data
    y_data = y.data
    plt.plot(x_data, y_data, z, **kwargs)

##################################
# main 실행 부분
# 데이터 로드 ( get_data )
# w, b 초기화 (학습 대상)
# 학습 루프 (simple_network --> loss --> w, b update)
# 결과 시각화
##################################
if __name__ == '__main__':
    # 1.
    train_x, train_y = get_data()
    print(f'Shape of train_x : {train_x.shape}')
    print(f'Shape of train_y : {train_y.shape}')

    #2.
    w, b = get_weights()
    print(f'initial weight : {w}')
    print(f'initial bias : {b}')

    #3. 학습 루프
    num_epochs = 1000
    lr = 1e-6
    loss_x = []
    loss_y = []
    for epoch in range(num_epochs):
        # 가설 함수 예측 값
        y_pred = simple_network(train_x, w, b)

        # loss 값 구하고 미분 한번 수행
        loss = loss_func(train_y, y_pred)
        loss_x.append(epoch)
        loss_y.append(loss)
        # loss 값 체크
        if epoch % 100 == 0:
            print(f"Epoch : {epoch}, loss : {loss:.4f}, w : {w.data}, b : {b.data}")
        # param update : 최적의 기울기, y 절편을 찾아 가는 것
        update_param(lr)

    # 결과 시각화
    plot_variable(train_x, train_y, 'ro', label='Data')
    plot_variable(train_x, y_pred, label='Fitted Line')
    plt.legend()
    plt.title("linear Regression Fitting Result")
    plt.show()

    # loss 결과
    plt.plot(loss_x, loss_y)
    plt.title("Loss over the Epochs")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()