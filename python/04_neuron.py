class Neuron:
    def __init__(self):
        self.w = 1.0                                    # 가중치 w 초기화하기
        self.b = 1.0                                    # 절편 b 초기화하기
    
    def fwrd(self, x):
        y_hat = x * self.w + self.b                     # 선형 방정식으로 계산하기
        return y_hat
    
    def bwrd(self, x, err):
        w_grad = x * err                                # 가중치 w에 대한 gradient 계산
        b_grad = 1 * err                                # 절편 b에 대한 gradient 계산
        return w_grad, b_grad
    
    def fit(self, x, y, epochs = 100):
        for i in range(epochs):                         # Epoch의 크기만큼 반복해라
            for x_i, y_i in zip(x, y):                  # 모든 샘플에 대해서 반복해라
                y_hat = self.fwrd(x_i)                  # 순전파 계산하고
                err = -(y_i - y_hat)                    # 오차를 계산하고
                w_grad, b_grad = self.bwrd(x_i, err)    # 역전파 계산하고
                self.w -= w_grad                        # 가중치 업데이트 해주고
                self.b -= b_grad                        # 절편 업데이트 해줘라
