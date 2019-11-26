class LogisticNeuron:
    
    def __init__(self):
        self.w = None                                # 입력 데이터의 특성이 많아 가중치를 미리 초기화 하지 않음
        self.b = None                                # 가중치는 나중에 입력 데이터를 보고 특성 개수에 맞게 결정
        
    def fwrd(self, x):
        z = np.sum(x * self.w) + self.b              # 고차원의 특성이니깐 np.sum()함수를 써서 선형 방정식으로 계산
        return z
    
    def bwrd(self, x, err):
        w_grad = x * err                             # 가중치에 대한 그레디언트 계산
        b_grad = 1 * err                             # 절편에 대한 그레디언트 계산
        return w_grad, b_grad
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))                     # 시그모이드 계산
        return a
    
    def fit(self, x, y , epochs = 100):
        self.w = np.ones(x.shape[1])                 # 가중치 초기화 (x 데이터 크기 만큼)
        self.b = 0                                   # 절편 초기화

        for i in range(epochs):                      # epoch의 크기 만큼 반복
            for x_i, y_i in zip(x, y):               # 모든 샘플에 대해 반복
                z = self.fwrd(x_i)                   # 순전파 계산
                a = self.activation(z)               # 활성화 함수 적용
                err = -(y_i - a)                     # 오차 계산
                w_grad, b_grad = self.bwrd(x_i, err) # 역전파 계산
                self.w -= w_grad                     # 가중치 업데이트
                self.b -= b_grad                     # 절편 업데이트 
                
    def predict(self, x):
        z = [self.fwrd(x_i) for x_i in x]            # 선형 함수 적용
        a = self.activation(np.array(z))             # 활성화 함수 적용
        return a > 0.5                               # 임계 함수
