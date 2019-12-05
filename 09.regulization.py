class SingleLayer_Reg:
    def __init__(self, learning_rate = 0.1, l1 = 0, l2 = 0):
        self.w = None
        self.b = None
        self.loss = []
        self.val_loss = []
        self.w_history = []
        self.lr = learning_rate
        self.l1 = l1
        self.l2 = l2
        
    def fwrd(self, x):
        z = np.sum(x * self.w) + self.b
        return z
    
    def activation(self, z):
        a = 1 / (1 + np.exp(-z))
        return a
    
    def bwrd(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    def reg_loss(self):
        return self.l1 * np.sum(np.abs(self.w)) + self.l2/2 * np.sum(self.w**2)
    
    def update_val_loss(self, x_val, y_val):
        if x_val is None:
            return
        val_loss = 0
        for i in range(len(x_val)):
            z = self.fwrd(x_val[i])
            a = self.activation(z)
            a = np.clip(a, 1e-10, 1-1e-10)
            val_loss += -(y_val[i]*np.log(a) + (1-y_val[i])*np.log(1-a))
        self.val_loss.append(val_loss/len(y_val) + self.reg_loss())
        
    def fit(self, x, y, epochs = 100, x_val = None, y_val = None):
        self.w = np.ones(x.shape[1])
        self.b = 0
        self.w_history.append(self.w.copy())
        np.random.seed(42)
        for i in range(epochs):
            loss = 0
            indexes = np.random.permutation(np.arange(len(x)))
            for i in indexes:
                z = self.fwrd(x[i])
                a = self.activation(z)
                err = -(y[i] - a)
                w_grad, b_grad = self.bwrd(x[i], err)
                w_grad += self.l1 * np.sign(self.w) + self.l2 * self.w
                self.w -= self.lr * w_grad
                self.b -= b_grad
                self.w_history.append(self.w.copy())
                a = np.clip(a, 1e-10, 1-1e-10)
                loss += - (y[i]*np.log(a) + (1-y[i])*np.log(1-a))
            self.loss.append(loss/len(y) + self.reg_loss())
            self.update_val_loss(x_val, y_val)
            
    def predict(self, x):
        z = [self.fwrd(xi) for xi in x]
        return np.array(z) > 0
    
    def score(self, x, y):
        return np.mean(self.predict(x)==y)