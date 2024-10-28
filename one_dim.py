import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_err(points, m, b):
    # MSE
    mse = (np.sum((points[: , 1] - (m*points[: , 0] + b))  ** 2 )) / len(points)
    return mse

def step_gradient(b_curr, m_curr, points, lr):
    
    
    x = points[:, 0]
    y = points[:, 1]
    theta = np.array([b_curr, m_curr])
    y_pred = theta[0] + theta[1] * x

    err = y - y_pred

    grad_theta0 = -2 * np.sum(err) / len(points)
    grad_theta1 = -2 * np.sum(x*err) / len(points)

    return theta[0] - lr*grad_theta0, theta[1] - lr*grad_theta1

    

def gd_runner(points, starting_b, starting_m, lr, num_iters):
    b , m = starting_b, starting_m
    for i in range(num_iters):
        # update b and m with new and more accurate b and m by performing GD
        b, m = step_gradient(b, m, points, lr)
    return b, m

# Linear Regression
def run():
    df = pd.read_csv('data.csv')
    points = df.to_numpy()
    print(points)

    # define hyperparams: 
    #   lr: how fast should our model converge
    lr = 0.0001

    # y = mx + b
    initial_b = 0
    initial_m = 0
    iter_num = 1000
        
    # train the model
    print(f'starting GD at b = {initial_b} , m={initial_m}, err={compute_err(points, initial_m, initial_b)}')
    x = points[:, 0]
    y = points[:, 1]
    plt.scatter(x, y, color='blue', label="Data points")
    plt.plot(x, initial_m * x + initial_b, color='green', label="Initial prediction line")
    
    b, m = gd_runner(points, initial_b, initial_m, lr, iter_num)
         
    print(f'ending GD at b = {b} , m={m}, err={compute_err(points, m, b)}')
    plt.plot(x, m * x + b, color='red', label="Final prediction line")

    # Add plot labels and legend
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    run()