class SingleLayer:

    def __init__(self):
        self.w = None
        self.b = None
        self.loss = []

    def fwrd(self, x):
        z =  np.sum(x * self.w) + self.b
        return z

    def bwrd(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad

    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a

    def fit(self, x, y, epochs = 100):
        self.w = np.ones(x.shape[1])
        self.b = 0

        for i in range(epochs):
            loss = 0
            # [4-1]: index 섞기
            indexes = np.random.permutation(np.arange(len(x))) # 인덱스를 섞음
            for i in indexes:
                z = self.fwrd(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.bwrd(x[i], err)
                self.w -= w_grad
                self.b -= b_grad
            # [4-2]: np.clip으로 주어진 밖의 값을 제거함
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += -(y[i] * np.log(a) + (1 - y[i]) * np.log(1 - a))
            # [4-3]: self.loss에 저장
            self.loss.append(loss/len(y))

    def predict(self, x):
        z = [self.fwrd(x_i) for x_i in x] # 정방향 계산
        return np.array(z) > 0 # 임계 함수 (계단 함수 적용)

    def score(self, x, y):
        return np.mean(self.predict(x) == y)