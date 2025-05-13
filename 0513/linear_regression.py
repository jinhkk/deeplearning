import torch
import random
import numpy as np
from astropy.io.misc.asdf.connect import asdf_identify

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
    w = torch.randn(1)
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
    y_pred = torch.matmul(x, w) + b

    return y_pred

##################################
# main 실행 부분
# 데이터 로드 ( get_data )
# w, b 초기화 (학습 대상)
# 학습 루프 (simple_network --> loss --> w, b update)
# 결과 시각화
##################################
if __name__ == '__main__':
    # 1.
    ㅅ