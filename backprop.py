import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import copy

from sklearn.metrics import plot_confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

def sigmoid(X):
    return 1. / (1. + np.exp(-X))

def dsigmoid(X):
    f = sigmoid(X)
    return f * (1 - f)

def plot_ims(imgs, labels):
    fig = plt.figure(figsize=(6, 10))
    grid = ImageGrid(fig, 111, nrows_ncols=(6, 4), axes_pad=0.7)

    for ax, (idx, im) in zip(grid, enumerate(imgs[:24])):
        ax.imshow(im, cmap='gray')
        ax.set_title(labels[idx])

def conv_to_imgs(x):
    img_list = []
    for img in x:
        b = img[:-1].copy()
        b.resize((8, 8))
        img_list.append(b)

    return img_list

def conv_to_onehot(y):
    onehot = []
    for label in y:
        onehot.append(np.array([0] * 10))
        onehot[-1][label] = 1

    return np.array(onehot)

def get_data():
    data = load_digits()
    X = copy.deepcopy(data["data"])
    y = copy.deepcopy(data["target"])
    
    X = np.array([np.append(np.array(x) / 16., np.array([1.])) for x in X])
    y = conv_to_onehot(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test

class MultiLayerNN(BaseEstimator, ClassifierMixin):
    def __init__(self, layers, maxiter=1000, lr=1e-3, epsilon=1e-4):
        self.lr = lr
        self.maxiter = maxiter
        self.epsilon = epsilon
        self.layers = layers
        self.theta = sigmoid
        self.dtheta = dsigmoid
        self.classes_ = list(range(11))

        self.W = []
        for i in range(len(layers) - 1):
            self.W.append(np.random.randn(layers[i+1], layers[i]))

    def forward_prop(self, X):
        V = [X]
        Y = [X]

        for i in range(len(self.W)):
            V.append(np.dot(self.W[i], Y[-1]))
            Y.append(self.theta(V[-1]))

        return Y, V

    def localgrad(self, y, Y, V):
        e = y - Y[-1]
        E = 0.5 * np.dot(e.T, e)
        n = len(Y)
        Delta = [0] * n
        Delta[n-1] = e * self.dtheta(V[n-1])

        for j in range(n-2, 0, -1):
            Delta[j] = self.dtheta(V[j]) * np.dot(self.W[j].T, Delta[j+1])

        return Delta, E

    def update_weights(self, Delta, Y):
        for i in range(len(self.W)):
            self.W[i] += self.lr * np.dot(Delta[i+1], Y[i].T)

    def fit(self, X, y):
        self.W = []
        for i in range(len(self.layers) - 1):
            self.W.append(np.random.randn(self.layers[i+1], self.layers[i]))

        for i in range(self.maxiter):
            GE = 0

            for j in range(X.shape[0]):
                Y, V = self.forward_prop(X[j].reshape(X[j].shape[0], 1))
                Delta, E = self.localgrad(y[j].reshape(y[j].shape[0], 1), Y, V)
                self.update_weights(Delta, Y)
                GE += E

            print("Iteration: {}, Global error: {}".format(i, GE))

            if GE < self.epsilon:
                break

    def predict(self, X):
        y_pred = []

        for x in X_test:
            Y, _ = self.forward_prop(x)
            y_pred.append(np.argmax(Y[-1]))

        return y_pred

X_train, X_test, y_train, y_test = get_data()

model = MultiLayerNN([65, 12, 12, 10], maxiter=2000)
model.fit(X_train, y_train)

y_true = []
y_pred = model.predict(X_test)

for y in y_test:
    y_true.append(np.argmax(y))

print(accuracy_score(y_true, y_pred))

imgs = conv_to_imgs(X_test.copy())
plot_ims(imgs, y_pred)
plot_confusion_matrix(model, X_test, y_true)
plt.show()

