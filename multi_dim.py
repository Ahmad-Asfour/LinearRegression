
class Multi_Dim_LR:
    def __init__(self, points, initial_theta, iternum, lr):
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

    def compute_err(self, points, theta):
        '''
        Computes and returns MSE for a given theta
        '''
        pass
    
    def gdb_runner(self, points, starting_theta, lr, iternum):
        '''
        goes through all iterations in iternum continuously updating theta
        returns theta
        '''
        pass

    def step_gradient(self, theta, points, lr):
        '''
        Calculates the next step by using the average gradient of a batch of datapoints
            -> basically for each datapoint in the batch we need to calculate each "feature's" (or "weight's") gradient
        '''
        pass

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

