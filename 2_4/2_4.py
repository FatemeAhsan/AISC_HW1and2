# In the name of Allah
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

col_name = '<CLOSE>'
df = pd.read_csv('project_data.csv', usecols=[col_name])
values_close = df[col_name].tolist()

x = range(1, len(values_close) + 1)

window_size = 50
means = [np.mean(values_close[i:i + window_size]) for i in range(len(values_close) - window_size)]

plt.plot(x, values_close)
plt.plot(range(len(means)), means)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Famelli Shares and its moving average')
plt.show()

