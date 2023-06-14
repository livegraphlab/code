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

def construct_graph_snapshot_with_nx(time, timeDict):
    time = time + ' 23:59:59'
    print(time)
    unixtime = int(pd.Timestamp(time).timestamp())

    G = nx.DiGraph()

    nodeDict = dict()
    idx2node = dict()

    idx = 0

    for k,v in timeDict.items():
        if k <= unixtime:
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

                G.add_nodes_from([fromID, toID])
                G.add_edges_from([(fromID, toID)])
        else:
            break
    
    nodes = G.number_of_nodes()
    edges = G.number_of_edges()
    connected = nx.is_connected(G)
   
    print('Number of nodes: ' + str(nodes))
    print('Number of edges: ' + str(edges))
    print('Is connected: ' + str(connected))
    print('-'*60)
    print()

    density = nx.density(G)
    reciprocity = nx.overall_reciprocity(G)
    assort = nx.degree_assortativity_coefficient(G)

    print('Density: ' + str(density))
    print('Reciprocity: ' + str(reciprocity))
    print('Assortativity: ' + str(assort))
    print('-'*60)
    print()
    return density, reciprocity, assort


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
    result, _ = G.GetClustCfAll()
    
    print('Number of nodes: ' + str(nodes))
    print('Number of edges: ' + str(edges))
    print('Is connected: ' + str(connected))
    print('All nodes: ' + str(len(nodeDict)))
    print('Effective diameter: ' + str(diameter))
    print('Average clustering aoefficient: ' + str(result[0]))
    print('Closed triads: ' + str(result[1]))
    print('Open triads: ' + str(result[2]))
    
    print('-'*60)
    print()

    return nodes, edges

def construct_date_range(start, end):
    s = datetime.datetime.strptime(start, "%Y-%m-%d")
    e = datetime.datetime.strptime(end, "%Y-%m-%d")
    
    end_labels = [x[:10] for x in pd.date_range(start, end, freq='Y').astype(np.str_)][0::1]

    return end_labels

def plot_properties(snapshots, diameters, figname):
    plt.figure()
    plt.xlabel('Year')
    # plt.xscale('log')
    # plt.xlim(0, max(snapshots)+1)

    xlabels = ['2017', '2018', '2019', '2020', '2021', '2022']
    plt.xticks([1,3,5,7,9,10], xlabels)
    plt.ylabel('Assortativity')
    # plt.yscale('log')
    plt.plot(snapshots, diameters, 'bs-')

    # e = [0.039193353230620315, 0.060927697177096854, 0.06300851667150303, 0.14543281453709644, 0.19343588769241843, 0.22960517043554982]
    # e = [0.024844720496894408,  0.12940166586486307, 0.09497841856908573, 0.09832508031594966, 0.05955088909430863, 0.058212097853274435]
    # e = [0.00625, 1.7577687980440596e-05, 3.994098070478745e-06, 2.894681898457132e-06, 1.986467207826068e-06, 1.2977164292389241e-06]
    e = [-0.3741475095697354, -0.3119950706759454, -0.2727098500781383, -0.16548620830164495, -0.060476653840105817, -0.04531796992084]

    # f = [0.060927697177096854, 0.14543281453709644, 0.22960517043554982]
    # f = [0.12940166586486307, 0.09832508031594966, 0.058212097853274435]
    # f = [1.7577687980440596e-05, 2.894681898457132e-06, 1.2977164292389241e-06]
    f = [-0.3119950706759454, -0.16548620830164495, -0.04531796992084]

    plt.plot([1,3,5,7,9,10], e, 'gx-')

    plt.plot([3,7,10], f, 'ro-')

    plt.legend(['Half a year', 'One year', 'Two years'])

    plt.savefig(figname, format="eps")


if __name__=='__main__':    
    f = open('timeDict.pkl', 'rb')
    timeDict = pickle.load(f)
    f.close()

    f = open('idx2node.pkl', 'rb')
    idx2node = pickle.load(f)
    f.close()
    
    start, end = '2017-07-12', '2022-08-01'
    # start, end = '2017-12-31', '2022-08-01'
    # start, end = '2018-01-01', '2018-12-31'

    ranges = construct_date_range(start, end)
    ranges.append(end)
    # ranges[-1] = end
    print(ranges)

    for i, time in enumerate(ranges):
    # for pretime, curtime in zip(ranges[:-1], ranges[1:]):
        print('Snapshot ' + str(i))
        # print('Snapshot...')
        nodes, edges = construct_graph_snapshot_with_snap(time, timeDict, idx2node)
        # construct_graph_snapshot_with_nx(time, timeDict)
