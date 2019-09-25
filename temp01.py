import numpy as np
import matplotlib.pyplot as plt

DEBUG = True
ALPHA = 0.01

## Rock/Paper/Scissors

A = np.array([[ 0, -1,  1],     # actions = R | P | S
              [ 1,  0, -1],
              [-1,  1,  0]])

## Stag Hunt

A = np.array([[4, 1],
              [3, 3]])

m, n = A.shape
B = A.T

x = np.random.dirichlet((1,)*m)
y = np.random.dirichlet((1,)*m)

trace_x, trace_y = [], []

for i in range(1000):
    dx = x * (x.dot(A) - x.dot(A).dot(y))
    dy = y * (y.dot(B) - y.dot(B).dot(x))

    x += ALPHA * dx
    y += ALPHA * dy

    # Normalize (due to rounding errors)
    x /= sum(x)
    y /= sum(y)

        print(i, ' x = ', x)
        print(i, ' y = ', y)

    trace_x.append(x.copy())
    trace_y.append(y.copy())

def generate_trace(A, n_steps=1000):
    m, n = A.shape
    B = A.T

    x = np.random.dirichlet((1,)*m)
    y = np.random.dirichlet((1,)*m)

    trace_x, trace_y = [], []

    for i in range(n_steps):
        dx = x * (x.dot(A) - x.dot(A).dot(y))
        dy = y * (y.dot(B) - y.dot(B).dot(x))

        x += ALPHA * dx
        y += ALPHA * dy

        # Normalize (due to rounding errors)
        x /= sum(x)
        y /= sum(y)

        if DEBUG:
            print(i, ' x = ', x)
            print(i, ' y = ', y)

        trace_x.append(x.copy())
        trace_y.append(y.copy())
    
    trace_x = np.array(trace_x)
    trace_y = np.array(trace_y)

    return trace_x, trace_y
        
trace_x, trace_y = generate_trace(A)

plt.plot(trace_x[:, 0], trace_y[:, 0])
plt.show()
