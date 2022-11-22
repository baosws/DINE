from matplotlib.lines import Line2D
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import matplotlib
import networkx as nx
from utils.metrics import F1, Precision, Recall

matplotlib.rc('font', family='DejaVu Sans')
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['lines.linewidth'] = 1
matplotlib.rcParams['text.latex.preamble'] = r'\boldmath'

pd.set_option("display.precision", 2)

def plot_results_imposed(A_pred, A_true):
    A_pred = A_pred.astype(bool)
    A_true = A_true.astype(bool)
    cmap = np.zeros(A_pred.shape, dtype=int)
    cmap[A_pred & A_true] = 1 # correct both directions
    cmap[A_pred & ~A_true] = 2 # extra edges
    cmap[A_true & ~A_pred] = 3 # missing edges
    G = nx.Graph()
    G.add_edges_from(np.array(np.nonzero(A_pred | A_true)).T)
    edge_colors = list(map(lambda x: ['red', 'green', 'blue'][cmap[x[0], x[1]] - 1], G.edges()))
    nx.draw_circular(G, with_labels=True, edge_color=edge_colors, font_color='white', width=2)
    custom_lines = [Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='green', lw=4),
                    Line2D([0], [0], color='blue', lw=4)]
    plt.gca().legend(custom_lines, ['Correct', 'Extra', 'Missing'])
    title = ', '.join([f'{score.__name__}: {score(A_true, A_pred):.1f}' for score in [F1, Precision, Recall]])
    print(title)
    plt.title(title)
    plt.show()