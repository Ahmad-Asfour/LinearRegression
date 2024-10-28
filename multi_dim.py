import numpy as np
class Multi_Dim_LR:
    def __init__(self, points, initial_theta, iternum, lr, batch_size):
        '''
        points: 2D numpy array (nxd)
        initial_theta: np array of length d+1
        iternum: int
        lr: float
        '''
        self.points = points
        self.initial_theta = initial_theta
        self.iternum = iternum
        self.lr = lr
        self.batch_size = batch_size

    def compute_err(self, points, theta):
        '''
        Computes and returns MSE for a given theta
        '''
        n = len(points)
        bias = theta[0]
        weights = theta[1:]

        mse = (np.sum((points[: , -1] - (points[: , :-1] @ weights + bias))  ** 2 )) / n

        return mse
    
    def gdb_runner(self, points, starting_theta, lr, iternum):
        '''
        goes through all iterations in iternum continuously updating theta
        returns theta
        '''
        theta = starting_theta
        for i in range(iternum):
            theta = self.step_gradient(theta, points, lr)
        return theta
    

    def step_gradient(self, theta, points, lr):
        '''
        Calculates the next step by using the average gradient of a batch of datapoints
            -> basically for each datapoint in the batch we need to calculate each "feature's" (or "weight's") gradient
        '''
        batch_size = self.batch_size
        X = points[: , :-1]
        y = points[: , -1:]

        beta = theta[0]
        weights = theta[1:]

        indices = np.random.choice(X.shape[0], size=batch_size, replace=False)
        x_batch = X[indices]
        y_batch = y[indices]

        y_pred = beta + x_batch @ weights
        err = y_batch - y_pred
        
        grad_beta = -2 * np.sum(err) / batch_size
        grad_weights = -2 * x_batch.T@err / batch_size

        new_weights = weights - lr*grad_weights
        new_beta = beta - lr*grad_beta
        new_theta = np.insert(new_weights, 0, new_beta)

        return new_theta




    def run(self):
        '''
        Runs LR
        '''
        print(f'starting GD at theta = {self.initial_theta} , err={self.compute_err(self.points, self.initial_theta)}')

        theta = self.gdb_runner(self.points, self.initial_theta, self.lr, self.iternum)

        print(f'starting GD at theta = {self.initial_theta} , err={self.compute_err(self.points, theta)}')
        
        

if __name__ == '__main__':
    '''
    
    '''
    pass

