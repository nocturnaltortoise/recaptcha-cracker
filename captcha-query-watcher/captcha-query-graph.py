import matplotlib.pyplot as plt
import numpy as np
import json
import os.path


def plot_graph(queries):
    X = np.arange(len(queries))
    plt.bar(X, queries.values(), align='center', width=0.5)
    plt.xticks(X, queries.keys())
    ymax = max(queries.values()) + 1
    plt.ylim(0, ymax)
    plt.show()


if os.path.isfile('queries.json'):
    with open('queries.json', 'r') as f:
        queries = json.load(f)

        plot_graph(queries)
else:
    print('Queries file does not exist, run captcha-watcher.')


