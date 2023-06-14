import numpy as np
import datetime
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict


filename = 'metadata.csv'

edgeDict = dict()

mintime = np.inf
maxtime = 0

repcnt = 0

with open(filename, 'r') as f:
    for line in f:
        contents = line.strip('\n\r').split(',')
        # from_address = contents[0]
        # to_address = contents[1]
        # unixtime = eval(contents[-1])
        from_address = contents[2]
        to_address = contents[3]
        # token_id = contents[4]
        unixtime = eval(contents[7])

        if from_address == to_address:
            repcnt += 1
            continue
        
        edge = (from_address, to_address)

        if edge not in edgeDict:
            edgeDict[edge] = [unixtime, unixtime]
        else:
            if edgeDict[edge][0] > unixtime:
                edgeDict[edge][0] = unixtime
            
            if edgeDict[edge][1] < unixtime:
                edgeDict[edge][1] = unixtime


binDict = SortedDict()

bicount = 0

for edge, time in edgeDict.items():
    from_address = edge[0]
    to_address = edge[1]

    edge_reverse = (to_address, from_address)

    if edge_reverse in edgeDict:
        time2 = edgeDict[edge_reverse]

        bicount += 1

        if from_address == to_address:
            bicount += 1

        if time[0] < time2[0]:
            span = [time[0], time2[1]]
        else:
            span = [time2[0], time[1]]

        s = datetime.datetime.fromtimestamp(span[0])
        e = datetime.datetime.fromtimestamp(span[1])
        intervals = (e-s).days

        if intervals not in binDict:
            binDict[intervals] = 1
        else:
            binDict[intervals] += 1

plt.ylabel('Cumulative ratio', fontsize=12)
plt.xlabel('Maximum time interval of bi-directional links (Days)', fontsize=12)

totalcnt = 0

keylist = []
valist = []

cnt0 = binDict[0] / 2
cnt100 = 0

for k,v in binDict.items():
    # print(k,v)
    v = v / 2
    totalcnt += v
    # plt.plot(k+1, v)
    keylist.append(k+1)
    valist.append(v)

    if k >= 100:
        cnt100 += v

print(totalcnt)
print(cnt0)
print(cnt0 / totalcnt)
print(cnt100 / totalcnt)
print('========')

# plt.plot(keylist, valist)
plt.plot(keylist, np.cumsum(valist)/np.cumsum(valist)[-1])
plt.vlines(90, 0, 1, colors = "r", linestyles = "dashed")
plt.text(100, 0.5, r'day=90', fontsize=15)

print(np.cumsum(valist)[0]/np.cumsum(valist)[-1])


# plt.legend(loc="upper right")
# plt.grid()
plt.rcParams.update({'font.size': 12})

# print(binDict.keys())
# print(binDict.values())

print(len(edgeDict))
print('====')
print(binDict[0])
print('====')
print(bicount / 2.0)
print('====')
print(repcnt)

plt.savefig('mutual-links-cumul.eps', format="eps")
