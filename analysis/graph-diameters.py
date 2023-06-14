import numpy as np
import networkx as nx
from sortedcontainers import SortedDict
import datetime
import pandas as pd
import snap
import pickle
from networkx.algorithms.distance_measures import diameter
import matplotlib.pyplot as plt


def construct_time_dict(filename):
    nodeDict = dict()
    idx2node = dict()
    timeDict = SortedDict()
    
    idx = 0

    with open(filename, 'r') as f:
        for line in f:
            l = line.strip('\n\r').split(',')
            from_addr = l[2]
            to_addr = l[3]

            if from_addr == '0x0000000000000000000000000000000000000000' and to_addr == '0x0000000000000000000000000000000000000000':
                continue

            if from_addr not in nodeDict:
                nodeDict[from_addr] = idx
                idx2node[idx] = from_addr
                idx += 1
            
            if to_addr not in nodeDict:
                nodeDict[to_addr] = idx
                idx2node[idx] = to_addr
                idx += 1
            
            fromID = nodeDict[from_addr]
            toID = nodeDict[to_addr]

            timestamp = eval(l[-1])

            if timestamp not in timeDict:
                timeDict[timestamp] = [(fromID, toID)]
            else:
                timeDict[timestamp].append((fromID, toID))
    
    keys = timeDict.keys()[:]
    min_time = keys[0]
    max_time = keys[-1]

    print('Start time: ' + str(pd.to_datetime(min_time, utc=True, unit='s')))
    print('  End time: ' + str(pd.to_datetime(max_time, utc=True, unit='s')))
    print()

    f = open('timeDict.pkl', 'wb')
    pickle.dump(timeDict, f)
    f.close()

    f = open('idx2node.pkl', 'wb')
    pickle.dump(idx2node, f)
    f.close()

    return timeDict, idx2node

def construct_graph_snapshot_with_snap(time, timeDict, idx2node):
    time = time + ' 23:59:59'
    print(time)
    unixtime = int(pd.Timestamp(time).timestamp())

    G = snap.TNGraph.New()

    nodeDict = dict()
    idxDict = dict()

    idx = 0
    mintcnt = 0
    descnt = 0

    for k,v in timeDict.items():
        if k <= unixtime:
            for elems in v:
                from_addr = idx2node[elems[0]]
                to_addr = idx2node[elems[1]]

                if from_addr == '0x0000000000000000000000000000000000000000':
                    # print('null addr')
                    continue
                
                if to_addr == '0x0000000000000000000000000000000000000000':
                    # print('null address')
                    continue
                
                if elems[0] not in nodeDict:
                    nodeDict[elems[0]] = idx
                    G.AddNode(idx)
                    idxDict[idx] = elems[0] 
                    idx += 1
            
                if elems[1] not in nodeDict:
                    nodeDict[elems[1]] = idx
                    G.AddNode(idx)
                    idxDict[idx] = elems[1]
                    idx += 1
            
                fromID = nodeDict[elems[0]]
                toID = nodeDict[elems[1]]

                G.AddEdge(fromID, toID)
        else:
            break
    
    nodes = G.GetNodes()
    edges = G.GetEdges()
    connected = G.IsConnected()
    diameter = G.GetAnfEffDiam(10)

    print('Number of nodes: ' + str(nodes))
    print('Number of edges: ' + str(edges))
    print('Is connected: ' + str(connected))
    print('All nodes: ' + str(len(nodeDict)))
    print('Effective diameter: ' + str(diameter))
    print('-'*60)
    print()

    return nodes, edges

def construct_date_range(start, end):
    s = datetime.datetime.strptime(start, "%Y-%m-%d")
    e = datetime.datetime.strptime(end, "%Y-%m-%d")
    
    end_labels = [x[:10] for x in pd.date_range(start, end, freq='Y').astype(np.str_)][0::1]

    return end_labels

def plot_graph_diameter(snapshots, diameters, figname):
    plt.figure()
    plt.xlabel('Year')

    xlabels = ['2017', '2018', '2019', '2020', '2021', '2022']
    plt.xticks(snapshots, xlabels)
    plt.ylabel('Efficient diameter')

    f = [1.9300138833494498, 5.943290813993335, 5.537680555093422, 5.519672165039134, 4.953446098354001, 4.722491560018798]

    plt.plot(snapshots, diameters, 'bs-')
    plt.plot(snapshots, f, 'ro-')
    plt.legend(['With Null address', 'W/O Null address'])
    plt.savefig(figname, format="eps")


if __name__=='__main__':
    f = open('timeDict.pkl', 'rb')
    timeDict = pickle.load(f)
    f.close()

    f = open('idx2node.pkl', 'rb')
    idx2node = pickle.load(f)
    f.close()

    start, end = '2017-07-12', '2022-08-01'

    ranges = construct_date_range(start, end)
    ranges.append(end)
    print(ranges)

    for i, time in enumerate(ranges):
        print('Snapshot ' + str(i))
        nodes, edges = construct_graph_snapshot_with_snap(time, timeDict, idx2node)

# time = [1,2,3,4,5,6]
# diameters = [2.2219096087122336, 2.3608067606240195, 2.0124577313491496, 2.5350805688119893, 2.842470818038966, 2.95700686894497]
# plot_graph_diameter(time, diameters, 'graph-diameter.eps')
