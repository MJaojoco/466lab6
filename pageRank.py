from os import error
import numpy as np
import pandas as pd
import sys
import time

def constructGraph(filename):
    cols = ['node1', 'value1', 'node2', 'value2', 'opt']
    data = pd.read_csv(filename, sep=',', header=None, names=cols)

    V = dict()
    for line in data.values:
        if line[0] not in list(V.keys()):
            V[line[0]] = list()
        if type(line[2]) == str:
            V[line[0]].append(line[2].strip('" '))
        else:
            V[line[0]].append(line[2])
    return V

def constructSnapGraph(filename):
    data = pd.read_csv(filename, sep='\t', comment='#', header=None)
    V = dict()
    for line in data.values:
        if line[0] not in list(V.keys()):
            V[line[0]] = list()
        V[line[0]].append(line[1])
    return V

def pageRank(graph, d, epsilon):
    # converge to 1
    pr = dict()
    pr[0] = dict()
    for node in graph.keys():
        pr[0][node] = 1 / len(graph.keys())
    
    r = 1
    # while r < iterations:
    stop = epsilon
    while stop >= epsilon:
        # calculate next set of pageRank values
        pr[r] = dict()
        for node in graph.keys():
            temp = 0
            for n in graph[node]:
                if n in graph.keys():
                    temp += (pr[r - 1][n] / len(graph[n]))
            temp *= d
            pr[r][node] = ((1 - d) * (1/len(graph.keys()))) + temp
        
        stop = 0
        for node in graph.keys():
            stop += (pr[r][node] - pr[r - 1][node])
        r += 1
    return pr

def pageRanki(graph, d, iterations):
    # converge to 1
    pr = dict()
    pr[0] = dict()
    for node in graph.keys():
        pr[0][node] = 1 / len(graph.keys())
    
    r = 1
    while r < iterations:
        # calculate next set of pageRank values
        pr[r] = dict()
        for node in graph.keys():
            temp = 0
            for n in graph[node]:
                if n in graph.keys():
                    temp += (pr[r - 1][n] / len(graph[n]))
            temp *= d
            pr[r][node] = ((1 - d) * (1/len(graph.keys()))) + temp
        
        stop = 0
        for node in graph.keys():
            stop += (pr[r][node] - pr[r - 1][node])
        r += 1
    return pr
        
def sortFunc(v, pr):
    return pr[len(pr.keys()) - 1][v]


def main(argv):
    if len(argv) != 4:
        print("usage: python3 pageRank.py <csvName> <iterations> <small/snap>")
        sys.exit()

    filename = argv[1]
    start = time.time()
    if argv[3] == 'small':
        graph = constructGraph(filename)
        end = time.time()
    else:
        graph = constructSnapGraph(filename)
        end = time.time()
    print(f'Read time: {end - start}')
    start = time.time()
    if argv[3] == 'small':
        pageRanks = pageRanki(graph, 0.85, int(argv[2]))
        end = time.time()
    else:
       pageRanks = pageRank(graph, 0.85, int(argv[2]))
       end = time.time()
    print(f'Processing time: {end - start}')
    print(f'iterations: {len(pageRanks.keys())}')
    v = list(graph.keys())
    v.sort(key=lambda x: pageRanks[len(pageRanks.keys()) - 1][x], reverse=True)
    count = 1
    for node in v:
        print(f'{count}:\t{node} with pageRank: {pageRanks[len(pageRanks) - 1][node]}')
        count += 1

if __name__ == '__main__':
    main(sys.argv)
