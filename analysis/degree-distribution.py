import numpy as np
import datetime
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict
import powerlaw
from scipy.optimize import curve_fit


def power_law(x, a, b):
    return a * np.power(x, b)

def plot_degree_distribution(G, figname):
    degrees = G.degree()
    degrees = dict(degrees)
    values = sorted(set(degrees.values()))
    hist = [list(degrees.values()).count(x) for x in values]

    # f = plt.figure()
    plt.figure()
    pars, cov = curve_fit(f=power_law, xdata=values, ydata=hist, p0=[0, 0], bounds=(-np.inf, np.inf))
    stdevs = np.sqrt(np.diag(cov))
    res = hist - power_law(values, *pars)

    plt.plot(values, power_law(values, *pars), linestyle='--', linewidth=2, color='black')

    plt.text(100, 10000, r'$y \sim x^{-1.54}$', fontsize=15)

    plt.xlabel('Degree')
    plt.xscale('log')
    plt.xlim(1, max(values))
    
    plt.ylabel('Frequency')
    plt.yscale('log')
    plt.ylim(1, max(hist))
    plt.scatter(values, hist, marker='+', c='red')

    plt.rcParams.update({'font.size': 12})
    plt.savefig(figname, format="eps")

filename = 'logs-erc721-merge-with-value.csv'

digraph = nx.DiGraph()

with open(filename, 'r') as f:
    for line in f:
        contents = line.strip('\n\r').split(',')
        from_address = contents[2]
        to_address = contents[3]
        # unixtime = eval(contents[-1])

        digraph.add_nodes_from([from_address, to_address])
        digraph.add_edges_from([(from_address, to_address)])

print('Number of nodes: ' + str(len(digraph.nodes())))
print('Number of edges: ' + str(len(digraph.edges())))

plot_degree_distribution(digraph, 'degree.eps')
