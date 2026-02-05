import numpy as np

import sklearn.linear_model, sklearn.preprocessing
import scipy.signal, scipy.ndimage


def spikes_to_traces(T, Y, Tn, win_sigma=5e-3, Dt=1e-3):
    #winN = int(win/Dt)

    N = int(np.max(Y))+1
    X = np.zeros((N, int(Tn/Dt)+1))
    bins = np.arange(0, (int(Tn/Dt)+2))*Dt - Dt/2.
    for i in range(N):
        X[i,:], _ = np.histogram(T[Y==i], bins)
    X = scipy.ndimage.gaussian_filter1d(X, sigma=win_sigma/Dt, axis=1) / Dt

    return X

def split_train_test(input_t, input_y, train_test_split=0.8):
    stim_ixs = np.random.permutation(np.array(np.arange(len(input_t)), dtype=int))
    train_ixs = stim_ixs[:int(train_test_split*len(input_t))]
    test_ixs = stim_ixs[int(train_test_split*len(input_t)):]
    N_train = len(train_ixs)
    N_test = len(test_ixs)

    input_t_train = input_t[train_ixs]
    input_y_train = input_y[train_ixs]

    input_t_test = input_t[test_ixs]
    input_y_test = input_y[test_ixs]

    return input_t_train, input_y_train, input_t_test, input_y_test

def collect_states(input_t, input_y, tau, X, winT=1, Dt=1e-3):
    N_states = len(input_t)
    N = X.shape[0]
    reservoir_states = np.zeros((N_states, N*int(winT/Dt)))
    labels = np.zeros(N_states)

    state = np.zeros((N, int(winT/Dt)))
    for inp_ix, (t0, lbl) in enumerate(zip(input_t, input_y)):
        n0 = int((t0+tau)/Dt)
        nn = n0+int(winT/Dt)
        state = X[:, n0:nn]

        reservoir_states[inp_ix,:] = np.reshape(state, (1, N*int(winT/Dt)))
        labels[inp_ix] = lbl

    return reservoir_states, labels


def scale_states(train_reservoir_states, test_reservoir_states):
    scaler = sklearn.preprocessing.StandardScaler()
    train_reservoir_states_scaled = scaler.fit_transform(train_reservoir_states)
    test_reservoir_states_scaled = scaler.fit_transform(test_reservoir_states)

    return train_reservoir_states_scaled, test_reservoir_states_scaled

def fit_lm(train_reservoir_states, train_labels):
    lm = sklearn.linear_model.LogisticRegression(penalty='l2', solver='liblinear')
    lm.fit(train_reservoir_states, train_labels)
    return lm

def score(input_t, input_y, T, Y, input_ixs=[], winT=1, tau0=-100e-3, tauN=200e-3, tau_step=1e-3, win_sigma=5e-3, Dt=1e-3, N_cross=1):

    input_t = np.array(input_t)
    input_y = np.array(input_y, dtype=int)
    T = np.array(T)
    Y = np.array(Y, dtype=int)

    classes = np.unique(input_y)
    N_rep = int(np.sum(input_y==classes[0]))
    taus = np.arange(int(tau0/Dt), int(tauN/Dt), int(tau_step/Dt))*Dt

    # exclude input neurons:
    for y in input_ixs:
        T = np.delete(T, Y==y)
        Y = np.delete(Y, Y==y)

    # generate traces:
    Tn = np.max(input_t)+taus[-1]+winT
    X = spikes_to_traces(T, Y, Tn, win_sigma=win_sigma, Dt=Dt)

    scores = np.zeros((len(taus), N_cross))
    for tau_ix, tau in enumerate(taus):
        for cross_ix in range(N_cross):
            # train-test split:
            input_t_train, input_y_train, input_t_test, input_y_test = split_train_test(input_t, input_y, 0.8)
    
            # collect states:
            train_reservoir_states, train_labels = collect_states(input_t_train, input_y_train, tau, X, winT=winT, Dt=Dt)
            test_reservoir_states, test_labels = collect_states(input_t_test, input_y_test, tau, X, winT=winT, Dt=Dt)

            train_reservoir_states, test_reservoir_states = scale_states(train_reservoir_states, test_reservoir_states)
            lm = fit_lm(train_reservoir_states, train_labels)

            score = lm.score(test_reservoir_states, test_labels)

            scores[tau_ix, cross_ix] = score

    return taus, scores
