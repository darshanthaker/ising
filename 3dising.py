import numpy as np
import collections
from scipy.stats import genlogistic

Neighbor = collections.namedtuple('Neighbor', 'num weight')

class Ising3D(object):

    def __init__(self, visible_size, hidden_size, visible_adj=None, hidden_adj=None):
        assert len(visible_size) == 2
        assert len(hidden_size) == 2
        if visible_adj is None or hidden_adj is None:
            print "RBM learning of weights not implemented yet. pass in weights"
            sys.exit(1)
        self.visible_adj = visible_adj
        self.hidden_adj = hidden_adj
        self.visible_size = visible_size
        self.hidden_size = hidden_size
        self.init_system()

    def reset(self):
        self.init_system()

    # All states initially 1.
    def init_system(self):
        self.visible_states = [[1 for j in range(self.visible_size[1])] \
            for i in range(self.visible_size[0])]
        self.hidden_states = [[1 for j in range(self.hidden_size[1])]  \
            for i in range(self.hidden_size[0])]

    def draw_visible(self):
        for i in range(self.visible_size[0]):
            row = ""
            for j in range(self.visible_size[1]):
                if self.visible_states[i][j] == 1:
                    row += "#"
                else:
                    row += " " 
            print row

    def simulate(self, num_iterations):
        logistic = genlogistic(1)
        for t in range(num_iterations):
            visible_to_update = list()
            hidden_to_update = list()
            for (k, v) in self.visible_adj.iteritems():
                s = self.visible_states[k[0]][k[1]]
                node_prob = logistic.cdf(sum([s * self.hidden_states[a.num[0]][a.num[1]] * a.weight for a in v]))
                if np.random.rand() <= node_prob:
                    visible_to_update.append(k)
            for (k, v) in self.hidden_adj.iteritems():
                s = self.hidden_states[k[0]][k[1]]
                node_prob = logistic.cdf(sum([s * self.visible_states[a.num[0]][a.num[1]] * a.weight for a in v]))
                if np.random.rand() <= node_prob:
                    hidden_to_update.append(k)
            for node in visible_to_update:
                self.visible_states[node[0]][node[1]] *= -1 # flip state
            for node in hidden_to_update:
                self.hidden_states[node[0]][node[1]] *= -1 # flip state

# Hidden positively correlated with odd columns and
# negatively correlated with even columns
def build_adjacency(visible_size, hidden_size):
    visible_adj, hidden_adj = dict(), dict()
    for h_i in range(hidden_size[0]):
        for h_j in range(hidden_size[1]):
            for v_i in range(visible_size[0]):
                for v_j in range(visible_size[1]):
                    if v_j % 2 == 0:
                        w = 3
                    else:
                        w = -3
                    h_neighbor = Neighbor(num=(h_i, h_j), weight=w)
                    v_neighbor = Neighbor(num=(v_i, v_j), weight=w)
                    if (v_i, v_j) not in visible_adj:
                        visible_adj[(v_i, v_j)] = [h_neighbor]
                    else:
                        visible_adj[(v_i, v_j)].append(h_neighbor)
                    if (h_i, h_j) not in hidden_adj:
                        hidden_adj[(h_i, h_j)] = [v_neighbor]
                    else:
                        hidden_adj[(h_i, h_j)].append(v_neighbor)
    return visible_adj, hidden_adj
                        

def main():
    visible_size = (4, 4)
    hidden_size = (1, 1)
    visible_adj, hidden_adj = build_adjacency(visible_size, hidden_size)
    ising = Ising3D(visible_size, hidden_size, visible_adj, hidden_adj)
    print "Initial State:"
    ising.draw_visible()
    print "-----------------"
    print "Final State:"
    ising.simulate(1000)
    ising.draw_visible()

main()
