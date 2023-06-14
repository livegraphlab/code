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


def increase_statis(time, collectDict):
    time = time + ' 23:59:59'
    print(time)
    unixtime = int(pd.Timestamp(time).timestamp())

    tokenDict = dict()
    collectionDict = dict()

    for k,v in collectDict.items():
        # for elem in v:
        if v[0][2] <= unixtime:
            if k[0] not in collectionDict:
                collectionDict[k[0]] = 1
            
            if k not in tokenDict:
                tokenDict[k] = 1
    
    print('Number of tokens: ' + str(len(tokenDict)))
    print('Number of collectiions: ' + str(len(collectionDict)))


def construct_date_range(start, end):
    s = datetime.datetime.strptime(start, "%Y-%m-%d")
    e = datetime.datetime.strptime(end, "%Y-%m-%d")
    
    end_labels = [x[:10] for x in pd.date_range(start, end, freq='Y').astype(np.str_)][0::1]

    return end_labels

start, end = '2017-07-12', '2022-08-01'
# start, end = '2017-12-31', '2022-08-01'
# start, end = '2018-01-01', '2018-12-31'

ranges = construct_date_range(start, end)
ranges.append(end)
# ranges[-1] = end
print(ranges)

f = open('collectDict.pkl', 'rb')
collectDict = pickle.load(f)
f.close()

for i, time in enumerate(ranges):
    print('Snapshot ' + str(i))
    increase_statis(time, collectDict)


def plot_token_collection_increase(snapshots, diameters, figname):
    plt.figure()
    plt.xlabel('Year')

    xlabels = ['2017', '2018', '2019', '2020', '2021', '2022']
    plt.xticks(snapshots, xlabels)
    plt.ylabel('Counts')
    plt.yscale('log')

    f = [16, 606, 2494, 6874, 24302, 97675]

    plt.plot(snapshots, diameters, 'bs-')

    plt.plot(snapshots, f, 'ro-')


    plt.legend(['#Tokens', '#Collections'])

    plt.savefig(figname, format="eps")


time = [1,2,3,4,5,6]
nodes = [179, 755023, 17044969, 19189251, 33435020, 78019866]

plot_active_ratio(time, nodes, 'token-increase.eps')
