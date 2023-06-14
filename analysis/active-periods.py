import numpy as np
import datetime
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict
import pickle

filename = 'metadata.csv'

edgeDict = dict()

nodeDict = dict()

mintime = np.inf
maxtime = 0

repcnt = 0

with open(filename, 'r') as f:
    for line in f:
        contents = line.strip('\n\r').split(',')
        from_address = contents[2]
        to_address = contents[3]
        # token_id = contents[4]
        unixtime = eval(contents[7])

        if from_address == to_address:
            repcnt += 1
            continue
        
        if from_address not in nodeDict:
            nodeDict[from_address] = [unixtime, unixtime, 1]
        else:
            if nodeDict[from_address][0] > unixtime:
                nodeDict[from_address][0] = unixtime
            
            if nodeDict[from_address][1] < unixtime:
                nodeDict[from_address][1] = unixtime
            
            nodeDict[from_address][2] += 1
        
        if to_address not in nodeDict:
            nodeDict[to_address] = [unixtime, unixtime, 1]
        else:
            if nodeDict[to_address][0] > unixtime:
                nodeDict[to_address][0] = unixtime
            
            if nodeDict[to_address][1] < unixtime:
                nodeDict[to_address][1] = unixtime
            
            nodeDict[to_address][2] += 1


binDict = SortedDict()

binDictW = SortedDict()

bicount = 0

maxcount = 0
tempnode = None

for node, elems in nodeDict.items():
    if elems[2] == 1:
        continue
    else:
        span = [elems[0], elems[1], elems[2]]
    
    s = datetime.datetime.fromtimestamp(span[0])
    e = datetime.datetime.fromtimestamp(span[1])
    # intervals = int((e-s).days / 7)
    v = span[2]

    intervals = (e-s).days

    if intervals not in binDict:
        binDict[intervals] = [1, v]
    else:
        binDict[intervals][0] += 1
        binDict[intervals][1] += v

print(binDict.keys())
print(len(binDict))

f = open('binDictValue.pkl', 'wb')
pickle.dump(binDict, f)
f.close()

def plot_active_ratio(snapshots, diameters, figname):
    plt.figure()
    plt.xlabel('Active duration (days)')

    plt.ylabel('Number of average transactions')
    diameters = list(diameters.values())[:40]
    res = []
    for elems in diameters:
        result = elems[1] / elems[0]
        res.append(result)
    plt.plot(snapshots, res, 'ro-')

    plt.savefig(figname, format="eps")


def plot_bar_graph(snapshots, diameter, figname):
    plt.figure()
    plt.yscale('log')
    plt.xlabel('Active duration (days)')

    plt.ylabel('Number of accounts')
    plt.bar(snapshots, diameter, width=0.5)

    plt.savefig(figname, format="eps")

f = open('binDictValue.pkl', 'rb')
binDict = pickle.load(f)
f.close()

time = list(range(1,41))
# plot_active_ratio(time, binDict, 'active-edge.eps')

# times = binDict.keys()
values = binDict.values()

plot_bar_graph(times, values[:40], 'active.eps')
