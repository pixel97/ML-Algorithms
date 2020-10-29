import numpy as np
import pandas as pd
import math
from sklearn.linear_model import LinearRegression


def predict_using_sklearn():
    data = pd.read_csv("scores.csv")
    model = LinearRegression()
    model.fit(data[['math']], data['cs'])
    return model.coef_, model.intercept_


def gradient_descent(x, y):
    m_curr = b_curr = 0
    iterations = 1000000
    learning_rate = 0.0002
    cost_previous = 0
    n = len(x)

    for i in range(iterations):
        y_pred = m_curr * x + b_curr
        cost = (1 / n) * sum([val**2 for val in (y - y_pred)])
        md = -(2 / n) * sum(x * (y - y_pred))
        bd = -(2 / n) * sum(y - y_pred)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost

        print("m{},b{},cost{},iteration{}".format(m_curr, b_curr, cost, i))
    return m_curr,b_curr


if __name__ == '__main__':
    data = pd.read_csv("scores.csv")
    x = np.array(data.math)
    y = np.array(data.cs)

    m, b = gradient_descent(x, y)
    print("Using gradient descent function: Coef {} Intercept {}".format(m, b))

    m_sklearn, b_sklearn = predict_using_sklearn()
    print("Using sklearn: Coef {} Intercept {}".format(m_sklearn, b_sklearn))


