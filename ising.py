import numpy as np
import sys
import collections
from scipy.stats import genlogistic

Neighbor = collections.namedtuple('Neighbor', 'num weight')

class Ising(object):

    def __init__(self, cols):
        self.adj, self.states = dict(), dict()
        self.cols = cols
        self.init_system()

    def reset(self):
        self.init_system()

    def __repr__(self):
        return " ".join([str(self.states[i]) for i in range(self.cols)])

    def add_edge(self, i, j, w):
        self.adj[i].append(Neighbor(num=j, weight=w))
        self.adj[j].append(Neighbor(num=i, weight=w))

    # System represented as an adjacency list
    def init_system(self):
        for i in range(self.cols):
            self.states[i] = 1 
            self.adj[i] = list()
        self.add_edge(0, 1, -50)
        self.add_edge(1, 2, 99)

    def simulate(self, num_iterations):
        logistic = genlogistic(1)
        for t in range(num_iterations):
            to_update = list()
            for (k, v) in self.adj.iteritems():
                s = self.states[k]
                node_prob = logistic.cdf(sum([s * self.states[a.num] * a.weight for a in v]))
                if np.random.rand() <= node_prob:
                    to_update.append(k)
            for node in to_update:
                self.states[node] *= -1 # flip state

    def get_energy(self):
        return -sum([self.states[k] * self.states[a.num] * a.weight for (k, v) in self.adj.iteritems() for a in v])/2.0
            
def main():
    ising = Ising(cols=3)
    print "Initial system energy: {}".format(ising.get_energy())
    ising.simulate(num_iterations=1000)
    print "Final system energy: {}".format(ising.get_energy())

main()
