import numpy as np
import scipy
import collections
import copy
import matplotlib.pyplot as plt
import seaborn

class Input:
    @staticmethod
    def out(X):
        return X
    @staticmethod
    def deriv(X):
        return X
    
class Sigma:
    @staticmethod
    def out(X):
        return 1 / (1 + np.exp(-X))
    @staticmethod
    def deriv(X):
        return X * (1 - X)

class Node:
    def __init__(self, id, type_node):
        self.id = id
        self.type_node = type_node
        self.input = 0
        self.out = 0
        self.ids_in = []
        self.error = 0
        self.deriv_act_f = 0
        self.grad = []

    def calc_out(self, X = 0):
        if X == 0:
            X = self.input
        self.out = self.type_node.out(X)
    
    def deriv(self, X = 0):
        if X == 0:
            X = self.out
        self.deriv_act_f = self.type_node.deriv(X)
        
    """def input(self, X):
        if self.w != [] :
            X_ = np.hstack((X, 1))
            return np.dot(X_, self.w.T)
        else:
           return X """
        

class Network:

    def __init__(self):
        self.nodes = []
        self.b = []
        self.w = np.array([])
        self.grad_w = np.array([])
        self.graph = collections.defaultdict(list)
        self.graphBack = collections.defaultdict(list)
        self.start = []
        self.end = []
        self.mu = 0
        self.total_error = list()
        
    def addNode(self, type_node):
        id = len(self.nodes)
        node = Node(id, type_node)
        self.nodes.append(node)
        return id
    
    def addConnect(self, id1, id2):
        self.graph[id1].append(id2)
        self.graphBack[id2].append(id1)
            
    def setDeepOfNodes(self):
        self.deepNodes = dict()

        visited = [False] * len(self.nodes)
        deep = 0
        for node in self.start:
            if visited[node] == False:
                self.deepNodes[node] = deep
                self.DeepOfNodes(node, visited, deep)
        
        self.Layers = collections.defaultdict(list)
        for item in self.deepNodes.items():
            self.Layers[item[1]].append(item[0])

        self.L = max(self.Layers.keys())
    
    def DeepOfNodes(self, v, visited, deep):
        visited[v] = True
        deep += 1
        copyV = copy.copy(self.graph[v])
        for neighbour in copyV:
            try:
                if self.deepNodes[neighbour] < deep:
                    self.deepNodes[neighbour] = deep

            except:
                self.deepNodes[neighbour] = deep

            if self.DeepOfNodes(neighbour, visited, deep) == True:
                return True
        return False
    
    def generateCoeffs(self):
        self.b = np.random.rand(self.L)
        self.w = np.zeros((len(self.graph), len(self.graph)))
        for i in np.arange(len(self.graph)):
            for j in self.graph[i]:
                self.w[i, j] = np.random.rand() * 0.5

                
    def forward(self, X):
        for l in np.arange(self.L + 1):
            for n in self.Layers[l]:
                if self.nodes[n].type_node == Input:
                    self.nodes[n].input = X[n]
                else:
                    inpt = []
                    w = []
                    for k in self.graphBack[n]:
                        inpt.append(self.nodes[k].out)
                        w.append(self.w[k, n])
                    self.nodes[n].input = np.dot(np.array(inpt), np.array(w).T) + self.b[l-1]

                self.nodes[n].calc_out()

    def predict(self, X):
        Y = []
        self.forward(X)
        for i in self.end:
            Y.append(self.nodes[i].out)
        return Y

    
    def backprop(self, X, y):
        self.grad_w = np.zeros((len(self.graph), len(self.graph)))
        total_error = np.array([])
        for i in np.arange(len(y)):
            #self.nodes[self.end[i]].error = - (y[i] - self.nodes[self.end[i]].out)
            err_i_quad = np.power(y[i] - self.nodes[self.end[i]].out, 2) * 0.5
            total_error = np.append(total_error, err_i_quad)
            
        total_error = np.sum(total_error)
        
        #cycle for backprop
        for l in np.arange(self.L, 0, -1):
            for i in np.arange(len(self.Layers[l])):
                n = self.Layers[l][i]
                cur_node = self.nodes[n]
                if l == self.L:
                    for k in self.graphBack[n]:
                        # dE_total / dOut
                        cur_node.error = -(y[i] - cur_node.out)
                        # dActiv_f / d_input =  cur_node.deriv_act_f
                        cur_node.deriv()
                        # dInput / dW
                        dInput_dw = self.nodes[k].out
                        self.grad_w[k, n] = cur_node.error * cur_node.deriv_act_f * dInput_dw

                else:
                    grad = 0
                    for k in self.graph[n]:
                        # dE_total / dOut
                        dEt_dOut = self.nodes[k].error
                        # dOut / dy
                        dOut_dy = self.nodes[k].deriv_act_f
                        # dy / dOut
                        dY_dOut = self.w[n, k]
                        grad += dEt_dOut * dOut_dy * dY_dOut
                    
                    # dActiv_f / d_input =  cur_node.deriv_act_f
                    cur_node.deriv()
                    for k in self.graphBack[n]:
                        # dInput / dW
                        dInput_dw = self.nodes[k].out
                        self.grad_w[k, n] = grad * cur_node.deriv_act_f * dInput_dw
                        #self.w[k, n] = self.w[k, n] - self.mu * self.grad_w[k, n]
            
        self.w = self.w - self.mu * self.grad_w
        return total_error
                            
           
    def adaptation_sgd(self, X, y, mu = 0.5):
        self.mu = mu
        X = np.array(X) 
        y = np.array(y) 
        
        #cycle for samples
        for ep in np.arange(1000):
            er_i = []
            for i in np.arange(X.shape[0]): 
                self.forward(X[i, :])
                er_i.append(self.backprop(X[i, :], y[i, :]))

            self.total_error.append(np.sum(er_i))
        
        #self.predict(np.array([1,2]))
        
        plt.plot(self.total_error)
        plt.show()
        a = 1
            #cycle for backprop
            #total_error = y - 1
            #for j in np.arange(self.L, -1, -1):
                
np.random.seed(30)            
    
nw       = Network()
idIn1    = nw.addNode(type_node=Input) 
idIn2    = nw.addNode(type_node=Input) 
idHL1    = nw.addNode(type_node=Sigma) 
idHL2    = nw.addNode(type_node=Sigma) 
idOut1   = nw.addNode(type_node=Sigma) 
idOut2   = nw.addNode(type_node=Sigma)

nw.start = [idIn1, idIn2]
nw.end   = [idOut1, idOut2]

nw.addConnect(idIn1, idHL1), nw.addConnect(idIn1, idHL2)
nw.addConnect(idIn2, idHL1), nw.addConnect(idIn2, idHL2)
nw.addConnect(idHL1, idOut1), nw.addConnect(idHL1, idOut2)
nw.addConnect(idHL2, idOut1), nw.addConnect(idHL2, idOut2)

nw.setDeepOfNodes()
nw.generateCoeffs()
X = np.array([[0.1,0.2]])#,
              #[0.3,0.4],
              #[0.5,0.6]])

y = np.array([[0.3,0.4]])#,
              #[0.5,0.6],
              #[0.7,0.8]])

nw.adaptation_sgd(X, y)


a = list()
a.append(Node(1, '1'))

