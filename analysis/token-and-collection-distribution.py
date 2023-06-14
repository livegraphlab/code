import numpy as np
import datetime
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sortedcontainers import SortedDict
import powerlaw
from scipy.optimize import curve_fit
import pickle
from scipy import stats


f = open('collectDict.pkl', 'rb')
collectDict = pickle.load(f)
f.close()

finalDict = dict()
for k,v in collectDict.items():
    # v.sort(key=takeThird)
    elems = v[-1][1]
    finalDict[k] = elems

userDict = dict()
for k,v in finalDict.items():
    if v not in userDict:
        userDict[v] = [k]
    else:
        userDict[v].append(k)

usercollectDict = dict()
usertokenDict = dict()

maxcnt = 0
collcnt = 0
idx = None

for k,v in userDict.items():
    temp = dict()
    for elem in v:
        if elem[0] not in temp:
            temp[elem[0]] = 1
    usercollectDict[k] = len(temp)
    usertokenDict[k] = len(v)

    if len(v) > maxcnt:
        maxcnt = len(v)
        collcnt = len(temp)
        idx = k

print('address: ' + str(idx))
print('tokens: ' + str(maxcnt))
print('collects: ' + str(collcnt))

values = sorted(set(usertokenDict.values()))
hist = [list(usertokenDict.values()).count(x) for x in values]

for k,v in usertokenDict.items():
    if v == values[-1]:
        print('top1 address: ' + str(k))
        print('top1 token: ' + str(v))
        print('top1 collections: ' + str(usercollectDict[k]))
    
    if v == values[-2]:
        print('top2 address: ' + str(k))
        print('top2 token: ' + str(v))
        print('top2 collections: ' + str(usercollectDict[k]))

print(values[:10])
print(np.cumsum(hist)[:10]/np.cumsum(hist)[-1])

values2 = sorted(set(usercollectDict.values()))
hist2 = [list(usercollectDict.values()).count(x) for x in values2]

print(values2[:10])
print(np.cumsum(hist2)[:10]/np.cumsum(hist2)[-1])

plt.figure()

plt.xlabel('Degrees')
plt.xscale('symlog')
plt.xlim(0, max(values) * 2)
    
plt.ylabel('Frequency')
plt.yscale('symlog')
plt.ylim(0, max(hist) * 2)
plt.scatter(values, hist, marker='.', c='blue')
plt.scatter(values2, hist2, marker='+', c='red')
# plt.scatter(values2, hist2, marker='x', c='green')
# plt.plot(values3, hist3, 'bx-')
plt.rcParams.update({'font.size': 12})
# print(pars)
plt.legend(['Holding token distribution', 'Holding collection distribution'])
figname = 'token-collection.eps'
plt.savefig(figname, format="eps")

