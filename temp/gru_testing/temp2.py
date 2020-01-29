"""
from "Advanced deep learning with Python" - chapter 7 
"""

import numpy as np 

# the first dimension represents the mini-batch
x = np.array([[0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])
y = np.array([3])

x = np.array([[0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0,
0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0]])
y = np.array([12])

def step(s, x, U, W):
    return x * U + s * W # desired result: U = W = 1

def forward(x, U, W):
    # number of samples in the mini-batch
    number_of_samples = len(x)

    # length of each sample
    sequence_length = len(x[0])

    # initialize the state activation for each sample along the sequence
    s = np.zeros((number_of_samples, sequence_length + 1))

    # update the state sover the sequence
    for t in range(0, sequence_length):
        s[:, t+1] = step(s[:, t], x[:, t], U, W)
    
    return s

def backward(x, s, y, W):
    sequence_length = len(x[0])

    # the network output is just the last activation of sequence
    s_t = s[:, -1]

    # compute the gradient of the output w.r.t. MSE loss function at final state
    gS = 2 * (s_t - y)

    # set the gradient accumulations to 0
    gU, gW = 0, 0

    # accumualte gradients backwards
    for k in range(sequence_length, 0, -1):
        # compute the parameter gradients and accumulate the results
        gU += np.sum(gS * x[:, k-1])
        gW += np.sum(gS * s[:, k-1])

        # compute the gradient at the output of the previous layer
        gS = gS * W
    
    return gU, gW

def train(x, y, epochs, learning_rate=0.0005):
    """Train the network"""

    # set initial parameters
    weights = (-2, 0) # (U, W)

    # Accumulate the losses and their respective weights
    losses = list()
    gradients_u = list()
    gradients_w = list()

    # Perform iterative gradient descent
    for i in range(epochs):
        # perform a forward and backward pass to get the gradients
        s = forward(x, *weights)#[0], weights[1])

        # compute the loss
        loss = (y[0] - s[-1, -1]) ** 2

        # store the loss and weights values for later display
        losses.append(loss)

        gradients = backward(x, s, y, weights[1])
        gradients_u.append(gradients[0])
        gradients_w.append(gradients[1])

        # update each parameter 'p' by p <- p - (gradient * learning_rate)
        # 'gp' is the gradient of parameter 'p'
        weights = tuple( (p - gp * learning_rate) 
                        for p, gp in zip(weights, gradients))
        
    print(weights)
    return np.array(losses), np.array(gradients_u), np.array(gradients_w)

def plot_training(losses, gradients_u, gradients_w):
    import matplotlib.pyplot as plt 

    # remove nan and inf values
    losses = losses[~np.isnan(losses)][:-1]
    gradients_u = gradients_u[~np.isnan(gradients_u)][:-1]
    gradients_w = gradients_w[~np.isnan(gradients_w)][:-1]

    # plot the weights U and W
    fig, ax1 = plt.subplots(figsize=(5, 3.4))
    ax1.set_ylim(-3, 20)
    ax1.set_xlabel('epochs')
    ax1.plot(gradients_u, label='grad U', color='blue',
    linestyle=':')
    ax1.plot(gradients_w, label='grad W', color='red', linestyle='--')
    ax1.legend(loc='upper left')
    # instantiate a second axis that shares the same x-axis
    # plot the loss on the second axis
    ax2 = ax1.twinx()
    # uncomment to plot exploding gradients
    ax2.set_ylim(-3, 10)
    ax2.plot(losses, label='Loss', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    losses, gradients_u, gradients_w = train(x, y, epochs=150)
    plot_training(losses, gradients_u, gradients_w)