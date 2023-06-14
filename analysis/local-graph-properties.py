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

def graph_snapshot_statistic(pretime, curtime, timeDict, idx2node):
    pretime = pretime + ' 23:59:59'
    print(pretime)
    preunixtime = int(pd.Timestamp(pretime).timestamp())
    preNodeDict = dict()

    curtime = curtime + ' 23:59:59'
    print(curtime)
    curunixtime = int(pd.Timestamp(curtime).timestamp())
    mintNodeDict = dict()

    newNodeDict = dict()
    nodeDict = dict()

    for k,v in timeDict.items():
        if k <= preunixtime:
            for elems in v:
                if elems[0] not in preNodeDict:
                    preNodeDict[elems[0]] = 1
                
                if elems[1] not in preNodeDict:
                    preNodeDict[elems[1]] = 1

                if elems[0] not in nodeDict:
                    nodeDict[elems[0]] = 1
                
                if elems[1] not in nodeDict:
                    nodeDict[elems[1]] = 1

        if k <= curunixtime and k > preunixtime:
            for elems in v:
                from_addr = idx2node[elems[0]]
                to_addr = idx2node[elems[1]]

                if elems[0] not in nodeDict:
                    nodeDict[elems[0]] = 1
                
                if elems[1] not in nodeDict:
                    nodeDict[elems[1]] = 1

                if from_addr == '0x0000000000000000000000000000000000000000':
                    if elems[1] not in preNodeDict:
                        mintNodeDict[elems[1]] = 1

    for k,v in timeDict.items():
        if k <= curunixtime and k > preunixtime:
            for elems in v:
                if elems[0] not in preNodeDict and elems[0] not in mintNodeDict:
                    newNodeDict[elems[0]] = 1
                    
                if elems[1] not in preNodeDict and elems[1] not in mintNodeDict:
                    newNodeDict[elems[1]] = 1

    print('Number of all nodes: ' + str(len(nodeDict)))
    print('Number of pre nodes: ' + str(len(preNodeDict)))
    print('Number of mint nodes: ' + str(len(mintNodeDict)))
    print('Number of new nodes: ' + str(len(newNodeDict)))


def graph_snapshot_statistic_edge(pretime, curtime, timeDict, idx2node):
    pretime = pretime + ' 23:59:59'
    print(pretime)
    preunixtime = int(pd.Timestamp(pretime).timestamp())
    preEdgeDict = dict()

    curtime = curtime + ' 23:59:59'
    print(curtime)
    curunixtime = int(pd.Timestamp(curtime).timestamp())
    
    edgeDict = dict()
    diEdgeDict = dict()
    biEdgeDict = dict()

    preNodeDict = dict()
    curnodeDict = dict()

    G_pre = snap.TNGraph.New()

    G_cur = snap.TNGraph.New()

    for k,v in timeDict.items():
        if k <= preunixtime:
            for elems in v:
                fromID = elems[0]
                toID = elems[1]
                if fromID not in preNodeDict:
                    G_pre.AddNode(fromID)
                    preNodeDict[fromID] = 1
                
                if toID not in preNodeDict:
                    G_pre.AddNode(toID)
                    preNodeDict[toID] = 1
                
                G_pre.AddEdge(fromID, toID)
        
        if k <= curunixtime:
            for elems in v:
                fromID = elems[0]
                toID = elems[1]

                if elems not in edgeDict:
                    edgeDict[elems] = 1

                if fromID not in curnodeDict:
                    G_cur.AddNode(fromID)
                    curnodeDict[fromID] = 1
                
                if toID not in curnodeDict:
                    G_cur.AddNode(toID)
                    curnodeDict[toID] = 1
                G_cur.AddEdge(fromID, toID)
    
    print('Total edges: ' + str(len(edgeDict)))
    print('for pre graph....')
    print('Self edges: ' + str(G_pre.CntSelfEdges()))
    print('Bidirectional edges: ' + str(G_pre.CntUniqBiDirEdges()))
    print('Directional edges: ' + str(G_pre.CntUniqDirEdges()))

    print('for cur graph....')
    print('Self edges: ' + str(G_cur.CntSelfEdges()))
    print('Bidirectional edges: ' + str(G_cur.CntUniqBiDirEdges()))
    print('Directional edges: ' + str(G_cur.CntUniqDirEdges()))


def count_new_old_nodes(pretime, curtime, timeDict):
    curtime = curtime + ' 23:59:59'
    print(curtime)
    curunixtime = int(pd.Timestamp(curtime).timestamp())

    pretime = pretime + ' 23:59:59'
    print(pretime)
    preunixtime = int(pd.Timestamp(pretime).timestamp())

    idx = 0

    o2ocntDict = dict()
    o2ncntDict = dict()
    n2ocntDict = dict()
    n2ncntDict = dict()

    nodeDict = dict()
    idx2node = dict()
    
    oldnodeDict = dict()
    curnodeDict = dict()

    edgeDict = dict()
    curEdgeDict = dict()

    for k,v in timeDict.items():
        if k <= preunixtime:
            for elems in v:
                if elems[0] not in oldnodeDict:
                    oldnodeDict[elems[0]] = idx
                    idx += 1
                
                if elems[1] not in oldnodeDict:
                    oldnodeDict[elems[1]] = idx
                    idx += 1
                
                if elems not in edgeDict:
                    edgeDict[elems] = 1
        
        if k > preunixtime and k <= curunixtime:
            for elems in v:
                if elems[0] in oldnodeDict and elems[1] not in oldnodeDict:
                    if elems[1] not in curnodeDict:
                        curnodeDict[elems[1]] = idx
                        idx += 1
                
                if elems[0] not in oldnodeDict and elems[1] in oldnodeDict:
                    if elems[0] not in curnodeDict:
                        curnodeDict[elems[0]] = idx
                        idx += 1
                
                if elems[0] not in oldnodeDict and elems[1] not in oldnodeDict:
                    if elems[0] not in curnodeDict:
                        curnodeDict[elems[0]] = idx
                        idx += 1

                    if elems[1] not in curnodeDict:
                        curnodeDict[elems[1]] = idx
                        idx += 1
                    
                if elems[0] in curnodeDict and elems[1] in oldnodeDict:
                    n2ocntDict[elems] = 1
                elif elems[0] in curnodeDict and elems[1] in curnodeDict:
                    n2ncntDict[elems] = 1
                elif elems[0] in oldnodeDict and elems[1] in curnodeDict:
                    o2ncntDict[elems] = 1
                elif elems[0] in oldnodeDict and elems[1] in oldnodeDict:
                    o2ocntDict[elems] = 1
                
                if elems not in edgeDict:
                    edgeDict[elems] = 1
                
                if elems not in curEdgeDict:
                    curEdgeDict[elems] = 1
    
    print('n2ocnt: ' + str(len(n2ocntDict)))
    print('n2ncnt: ' + str(len(n2ncntDict)))
    print('o2ocnt: ' + str(len(o2ocntDict)))
    print('o2ncnt: ' + str(len(o2ncntDict)))
    print('total edge cnt: ' + str(len(edgeDict)))
    print('current edge cnt: ' + str(len(curEdgeDict)))
    print('-'*50)


def construct_date_range(start, end):
    s = datetime.datetime.strptime(start, "%Y-%m-%d")
    e = datetime.datetime.strptime(end, "%Y-%m-%d")
    
    end_labels = [x[:10] for x in pd.date_range(start, end, freq='Y').astype(np.str_)][0::1]

    return end_labels

def plot_bar(snapshots, diameter, figname):
    plt.figure()
    plt.xlabel('Year')

    xlabels = ['2018', '2019', '2020', '2021', '2022']
    plt.xticks(snapshots, xlabels)

    plt.ylabel('Ratio')
    plt.bar(np.array(snapshots) - 0.2, diameter, width=0.4)
    plt.bar(np.array(snapshots) + 0.2, [0.3104, 0.1497, 0.3063, 0.7045, 0.1940], width=0.4)
    plt.legend(['edges with new nodes to/from olds(%)', 'edges between new nodes(%)'])

    plt.savefig(figname, format='eps')

def plot_line(snapshots, diameters, figname):
    plt.figure()
    plt.xlabel('Year')

    xlabels = ['2017', '2018', '2019', '2020', '2021', '2022']
    plt.xticks(snapshots, xlabels)
    plt.ylabel('Number of edges')
    plt.yscale('log')
    plt.plot(snapshots, diameters, 'bs-')

    # e = [6857, 23600 - 6859, 50882-23600, 271184- 50882, 775452 - 271184]
    e = [71483, 263035, 189172, 1083781, 1629049]

    # f = [156, 617 - 156, 1759 - 617, 14361 - 1759, 28500 - 14361]
    f = [6016, 12041, 56042, 459494, 760746]

    plt.plot(snapshots[1:], e, 'gx-')
    plt.plot(snapshots[1:], f, 'ro-')

    plt.legend(['#nodes', '#mint nodes added', '#non-mint nodes added'])

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
    # ranges[-1] = end
    print(ranges)

    for i, time in enumerate(ranges):
    # for pretime, curtime in zip(ranges[:-1], ranges[1:]):
        print('Snapshot ' + str(i))
        count_new_old_nodes(pretime, curtime, timeDict)
        # graph_snapshot_statistic_edge(pretime, curtime, timeDict, idx2node)
