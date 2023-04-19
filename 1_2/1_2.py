# In the name of Allah
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

col_name = '<CLOSE>'
df = pd.read_csv('project_data.csv', usecols=[col_name])
values_close = df[col_name].tolist()


def derivative(x, y):
    x = np.array(x)
    y = np.array(y)
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]

    ydot = y.copy()
    ydot[1:] = dy / dx
    ydot[0] = ydot[1]
    return ydot


x = range(1, len(values_close) + 1)
ydot = derivative(x, values_close)

plt.plot(x, values_close)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Famelli Shares')
plt.show()

plt.plot(x, ydot)
plt.xlabel('x')
plt.ylabel('ydot')
plt.title('Famelli Shares Derivative')
plt.show()

sets = [values_close[i:i + 10] for i in range(len(values_close) - 10)]
sets_predict = [sets[i + 1][9] - sets[i][9] > 0 for i in range(len(sets) - 1)]
print(sets)
print(sets_predict)