import numpy as np
import datetime
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import pickle
from scipy import stats


def construct_graph_snapshot_with_nx(pretime, curtime, timeDict):
    pretime = pretime + ' 23:59:59'
    print(pretime)
    preunixtime = int(pd.Timestamp(pretime).timestamp())

    curtime = curtime + ' 23:59:59'
    print(curtime)
    curunixtime = int(pd.Timestamp(curtime).timestamp())

    G_pre = nx.DiGraph()

    G_cur = nx.DiGraph()

    nodeDict = dict()
    idx2node = dict()

    idx = 0
    idxc = 0

    nodeDictc = dict()
    idx2nodec = dict()

    for k,v in timeDict.items():
        if k <= preunixtime:
            for elems in v:
                if elems[0] not in nodeDict:
                    nodeDict[elems[0]] = idx
                    idx2node[idx] = elems[0] 
                    idx += 1
            
                if elems[1] not in nodeDict:
                    nodeDict[elems[1]] = idx
                    idx2node[idx] = elems[1]
                    idx += 1
            
                fromID = nodeDict[elems[0]]
                toID = nodeDict[elems[1]]

                G_pre.add_nodes_from([fromID, toID])
                G_pre.add_edges_from([(fromID, toID)])
        
        if k <= curunixtime:
            for elems in v:
                if elems[0] not in nodeDictc:
                    nodeDictc[elems[0]] = idxc
                    idx2nodec[idxc] = elems[0] 
                    idxc += 1
            
                if elems[1] not in nodeDictc:
                    nodeDictc[elems[1]] = idxc
                    idx2nodec[idxc] = elems[1]
                    idxc += 1
            
                fromID = nodeDictc[elems[0]]
                toID = nodeDictc[elems[1]]

                G_cur.add_nodes_from([fromID, toID])
                G_cur.add_edges_from([(fromID, toID)])
        else:
            break

    return G_pre, G_cur

def power_law(x, a, b):
    return a * np.power(x, b)

def plot_degree_distribution(G_pre, G_cur, figname):
    degrees3 = G_pre.degree()
    degrees3 = dict(degrees3)
    values3 = sorted(set(degrees3.values()))
    hist3 = [list(degrees3.values()).count(x) for x in values3]

    vkDict = dict()

    for val in values3:
        for k,v in degrees3.items():
            if v == val:
                if v not in vkDict:
                    vkDict[v] = [k]
                else:
                    vkDict[v].append(k)
    degrees = G_cur.degree()
    degrees = dict(degrees)
    values = []
    for val in values3:
        result = vkDict[val]
        sumval = 0
        for ele in result:
            sumval += degrees[ele]
        values.append(sumval)

    plt.figure()

    plt.xlabel('Node Degrees')
    plt.xscale('symlog')
    plt.xlim(0, max(values3) * 2)
    
    plt.ylabel('Node Counts')
    plt.yscale('symlog')
    plt.ylim(0, max(values) * 2)
    plt.plot(values3, hist3, 'bx-')
    plt.plot(values3, np.array(values) - np.array(hist3), 'ro-')
    print('Pearson Correlation: ' + str(np.corrcoef(np.array(values), np.array(hist3))))
    print('Pearson correlation coefficient: ' + str(stats.pearsonr(np.array(values), np.array(hist3)).statistic))

    plt.rcParams.update({'font.size': 12})
    plt.legend(['Node Degrees in 2018', '#New nodes connected in 2019'])
    plt.savefig(figname, format="eps")
 

f = open('timeDict.pkl', 'rb')
timeDict = pickle.load(f)
f.close()

pretime = '2018-12-31'
curtime = '2019-12-31'

G_pre, G_cur = construct_graph_snapshot_with_nx(pretime, curtime, timeDict)

plot_degree_distribution(G_pre, G_cur, 'hub-node-2018.eps')
