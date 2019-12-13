class SingleLayer_tradeoff:
    def __init__(self, learning_rate = 0.1):
        self.w = None                                           # 가중치 초기화
        self.b = None                                           # 절편 초기화
        self.loss = []                                          # 평균 손실 저장 리스트
        self.val_loss = []                                      # 검증 세트 저장 리스트
        self.w_history = []                                     # 가중치 저장 리스트
        self.lr = learning_rate                                 # 학습률
    
    def fwrd(self, x):
        z = np.sum(x * self.w) + self.b                         # 직선 방정식
        return z
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))                                # 시그모이드
        return a
    
    def bwrd(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def update_val_loss(self, x_val, y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.fwrd(x_val[i])
            a = self.activation(z)
            a = np.clip(a, 1e-10, 1-1e-10)
            val_loss += -(y_val[i]*np.log(a) + (1-y)*np.log(1-a))
        self.val_loss.append(val_loss / len(y_val))
        
    def fit(self, x, y, epochs = 100, x_val = None, y_val = None):
        self.w = np.ones(x.shape[1])                            # 가중치 초기화
        self.b = 0                                              # 절편 초기화
        self.w_history.append(self.w.copy())                    # 가중치 기록
        np.random.seed(42)                                      # 무작위로 시드 지정
        for i in range(epochs):                                 # epoch 크기 만큼 학습 반복
            loss = 0                                            # 손실 초기화
            indexes = np.random.permutation(np.arange(len(x)))  # 인덱스 섞기
            for i in indexes:                                   # 모든 샘플에 대해 반복
                z = self.fwrd(x[i])                             # 정방향 계산
                a = self.activation(z)                          # 활성화 함수 적용
                err = -(y[i] - a)                               # 오차 계산
                w_grad, b_grad = self.bwrd(x[i], err)           # 역방향 계산
                self.w -= self.lr * w_grad                      # 가중치 업데이트
                self.b -= b_grad                                # 절편 업데이트
                self.w_history.append(self.w.copy())            # 가중치 기록
                a = np.clip(a, 1e-10, 1-1e-10)                  # 안전한 로그 계산을 위한 클리핑
                loss += -(y[i]*np.log(a) + (1-y)*np.log(1-a))   # 손실 함수 업데이트
            self.loss.append(loss / len(y))                     # epoch 마다 평균 손실 기록
            self.update_val_loss(x_val, y_val)                  # 검증 세트에 대한 손실 계산
                
    def predict(self, x):
        z = [self.fwrd(xi) for xi in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x) == y)