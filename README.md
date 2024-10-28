# Linear Regression from Scratch

This repository contains two implementations of linear regression from scratch in Python:
1. **Multi-dimensional Linear Regression** (`multi_dim.py`): This script implements batch gradient descent for multi-dimensional data, with a 3D plot to visualize results for two features and a target.
2. **Single-dimensional Linear Regression** (`single_dim.py`): This script demonstrates linear regression for a single feature, using gradient descent and visualizing the regression line.

## Files

### 1. `multi_dim.py`

This file contains a multi-dimensional linear regression model implemented from scratch using batch gradient descent. It can handle datasets with multiple features, and it includes functionality to visualize results in 3D when using two features.

#### Key Components:
- **Class `Multi_Dim_LR`**: Implements the multi-dimensional linear regression model.
  - `__init__`: Initializes parameters like the dataset (`points`), initial weights (`initial_theta`), learning rate (`lr`), number of iterations (`iternum`), and batch size (`batch_size`).
  - `compute_err`: Computes Mean Squared Error (MSE) for a given set of parameters.
  - `gdb_runner`: Executes batch gradient descent over a specified number of iterations.
  - `step_gradient`: Computes the gradients and updates the weights and bias for a batch of data points.
  - `plot_result`: Plots the regression plane and original data points in 3D (works for two features).
  - `run`: Runs the gradient descent, outputs initial and final errors, and plots the regression plane.

#### Running `multi_dim.py`:
1. The script generates a synthetic dataset with 3 features and a target variable using a known linear relationship plus noise.
2. The initial parameters are set to zero, and gradient descent is run for 1000 iterations with a learning rate of 0.01.
3. After training, it outputs the final model parameters and a 3D plot (for two features) of the fitted plane and original data.

#### Example Command:
To run the file:
```bash
python3 multi_dim.py
