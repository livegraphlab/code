# Continuous Subgraph Matching
For CSM frameworks such as `sj-tree`, `graphflow`, `iedyn`, `turboflux` and `symbi`, we use the source code provided by Xibo Sun et al, which is available at [here](https://github.com/RapidsAtHKUST/ContinuousSubgraphMatching). For RapidFlow, we use the source code provided by the original author, which can be found at [here](https://github.com/shixuansun/RapidFlow).

## Example
After a successful compilation, we can execute CSM using the following command.
```
./csm -q Q_1 -d data-y2021.graph -u insertion-y2022.graph -a graphflow --time-limit 3600
```
where `-a` means algorithms, and we can chosen from `sj-tree`, `graphflow`, `iedyn`, `turboflux`, and `symbi`.

Similarly, we execuate RapidFlow with the following command.
```
./RapidFlow.out -d data-y2021.graph -q Q_1 -u insertion-y2022.graph -num 1 -time_limit 3600
```