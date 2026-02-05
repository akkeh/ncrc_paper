import numpy as np


def load_mnist():
    import keras.datasets.mnist

    (train_X, train_y), (test_X, test_y) = keras.datasets.mnist.load_data()
    return train_X, train_y, test_X, test_y

def img_to_trains(X, N, Tn, Dt, max_freq):
    Xm, Xn = np.shape(X)
    Nt = int(Tn / Dt)

    It = np.array([], dtype=float)
    Ic = np.array([], dtype=int)
    for n in range(Nt):
        x = X[:, int(Xm/Nt*n)]
        for i in range(N):
            m = int(Xn/N*i)
            if np.random.rand() > (1-max_freq*Dt*x[m]):
                Ic = np.append(Ic, i)
                It = np.append(It, n*Dt)

    return It, Ic

def select_input_neurons(X, Y, x0, y0, fraction=0.3):
    dists = np.sqrt(np.power(X-x0, 2) + np.power(Y-y0, 2) )
    input_ixs = np.argsort(dists)[:int(fraction*len(X))]
    return input_ixs

def gen_inputs(digits, input_neurons, N_rep, t0=30, Dt=1e-3, input_len=50e-3, inter_input_interval_min=5, inter_input_interval_max=10, max_freq=500):

    classes = digits
    N_classes = len(classes)

    N_neurons = len(input_neurons)

    input_y = []
    input_t = []
    It = []
    Ic = []

    mnist_X, mnist_y, _, _ = load_mnist()

    input_y = np.random.permutation(list(classes)*N_rep)
    time = t0
    for input_n, input_class in enumerate(input_y):
        input_t.append(time)

        X = np.random.permutation(mnist_X[mnist_y == input_class])[0]
        X = X / np.sum(X) * 100

        block_t, block_c = img_to_trains(X, N_neurons, input_len, Dt, max_freq)


        for t, c in zip(block_t, block_c):
            It.append(time+t)
            Ic.append(input_neurons[c])

        time += inter_input_interval_min + (inter_input_interval_max-inter_input_interval_min)*np.random.rand()


    return input_y, input_t, It, Ic
