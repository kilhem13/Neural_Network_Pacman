# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 14:57:00 2019

Les indices des elements des matrice sont souvent opposés à ceux des demonstrations du
cours car j'utilise ici des vecteurs en ligne et non en colone.
"""

import numpy as np
import matplotlib.pyplot as matplot


def ReadData():
    x0 = []
    x1 = []
    y = []

    fichier = "Pacdata"
    x0, x1, x2, x3, y = np.loadtxt(fichier, delimiter=" ", unpack=True)

    x = np.c_[x0, x1, x2, x3]

    #matplot.scatter(x[:, 0], x[:, 1], c=y)
    y = np.reshape(y, (len(y), 1))
    #y_tab = np.ndarray(shape=(len(y), 4))
    y_tab = np.ndarray(shape=(len(y), 4))
    for i in range(len(y)):
        if y[i] == 0:
            y_tab[i][0], y_tab[i][1], y_tab[i][2], y_tab[i][3] = 1, 0, 0, 0
        if y[i] == 1:
                y_tab[i][0], y_tab[i][1], y_tab[i][2], y_tab[i][3] = 0, 1, 0, 0
        if y[i] == 2:
            y_tab[i][0], y_tab[i][1], y_tab[i][2], y_tab[i][3] = 0, 0, 1, 0
        if y[i] == 3:
            y_tab[i][0], y_tab[i][1], y_tab[i][2], y_tab[i][3] = 0, 0, 0, 1

    return x, y_tab


def InitializeNetwork(x):
    x_barre = np.c_[np.ones((len(x), 1)), x]

    N = len(x[0])  # features
    K = 9  # hiddien neurons
    J = 4  # output neurons

    # random intialisation of the paramaters of iputs neurons
    V = np.random.rand(K, N + 1)
    # random intialisation of the paramaters of outputs neurons
    W = np.random.rand(J, K + 1)

    return x_barre, V, W


def ActiveFunction(sigma):
    return np.reciprocal(1 + np.exp(-sigma))


def ForwardPropagation(x_barre, V, W):
    x_barrebarre = np.dot(x_barre, V.T)
    F = ActiveFunction(x_barrebarre)
    F_barre = np.c_[np.ones((len(F), 1)), F]
    F_barrebarre = np.dot(F_barre, W.T)

    G = ActiveFunction(F_barrebarre)

    return F, F_barre, G


def PartialDerivate(V, W, G, F_barre, F, Y, X_barre):
    dEdv = np.zeros((V.shape[0], V.shape[1]))

    for k in range(V.shape[0]):
        for n in range(V.shape[1]):
            for I in range(G.shape[0]):
                for j in range(G.shape[1]):
                    #print((Y[I][j]))
                    dEdv[k][n] = ((G[I][j] - Y[I][j]) * (G[I][j]) * (1 - G[I][j]) * W[j][k] * (
                                F[I][k] * (1 - F[I][k]) * X_barre[I][n]))

    dEdw = np.zeros((W.shape[0], W.shape[1]))
    for j in range(W.shape[0]):
        for k in range(W.shape[1]):
            for I in range(len(G)):
                dEdw[j][k] = ((G[I][j] - Y[I][j]) * G[I][j] * (1 - G[I][j]) * F_barre[I][k])

    return dEdv, dEdw


def BackwardPropagation(V, W, G, F_barre, F, Y, X_barre):
    # Backward propagation using BGD

    alpha1 = 0.1
    alpha2 = 0.1
    dEdv, dEdw = PartialDerivate(V, W, G, F_barre, F, Y, X_barre)
    W = W - alpha1 * dEdw
    V = V - alpha2 * dEdv

    return V, W


def SSE(y, g):
    E = (1 / 2) * np.sum(np.square(y - g))
    return E


def Question4():
    X, Y = ReadData()
    X_barre, V, W = InitializeNetwork(X)

    iteration = 1000
    cost_history = np.zeros((iteration, 1))

    for i in range(iteration):
        F, F_barre, G = ForwardPropagation(X_barre, V, W)
        V, W = BackwardPropagation(V, W, G, F_barre, F, Y, X_barre)
        cost_history[i] = SSE(Y, G)

    print(G)
    print("result")
    print(W, V)

    X_test = np.array([[1, 0, 0, 0], [1, 1, 0, 0],[0, 1, 1, 0], [0, 0, 0, 1]])
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    print(X_test)
    F_test, F_barre_test, G_test = ForwardPropagation(X_test, V, W)
    print("G")
    print(G_test)

    # cost plot
    matplot.figure()
    matplot.plot(np.array(range(0, iteration)), cost_history)
    matplot.title("SSE")
    matplot.show()

    matplot.figure()
    matplot.plot(G[:, 0], c="red")
    matplot.plot(G[:, 1], c="blue")
    matplot.plot(G[:, 2], c="red")
    matplot.plot(G[:, 3], c="blue")
    matplot.title("Classification")
    matplot.show()
Question4()
