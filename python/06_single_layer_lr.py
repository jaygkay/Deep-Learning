class SingleLayerLR:
    def __init__(self, learning_rate = 0.1):
        self.w = None                                                 # initiate a weight
        self.b = None                                                 # initiate a bias
        self.loss = []                                                # append a loss
        self.w_history = []                                           # append learning_rate
        self.lr = learning_rate

    def fwrd(self, x):
        z = np.sum(x * self.w + self.b)
        return z

    def bwrd(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad

    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a

    def fit(self, x, y, epochs = 100):
        self.w = np.ones(x.shape[1])                                  # initiate a weight
        self.b = 0                                                    # initiage a bias
        self.w_history.append(self.w.copy())                          # record the weight

        for i in range(epochs):                                       # iterate the learning as many as the size of the epochs
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))        # shuffle the index
            for i in indexes:                                         # interate for all samples
                z = self.fwrd(x[i])                                   # forward
                a = self.activation(z)                                # activation
                err = - (y[i] - a)                                    # compute the error
                w_grad, b_grad = self.bwrd(x[i], err)                 # backward
                self.w -= self.lr * w_grad                            # update the weight
                self.b -= b_grad                                      # update the bias
                self.w_history.append(self.w.copy())                  # record the wieght
                a = np.clip(a, 1e-10, 1-1e-10)                        # range setting for a safe log calculation
                loss += -(y[i] * np.log(a) + (1-y[i]) * np.log(1- a)) # update the loss
            self.loss.append(loss / len(y))                           # record loss every epoch

    def predict(self, x):
        z = [self.fwrd(xi) for xi in x]
        return np.array(z) > 0

    def score(self, x, y):
        return np.mean(self.predict(x)==y)