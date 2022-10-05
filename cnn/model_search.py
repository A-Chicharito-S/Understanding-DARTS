import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in PRIMITIVES:
            # primitive is a string for a certain operation, e.g., 'max_pool_3x3'
            op = OPS[primitive](C, stride, False)
            # OPS is a dictionary from "operations.py", whose key is the name of an operation, e.g., 'max_pool_3x3',
            # and whose value is the neural implementation of that operation
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
                # add Batch Normalization for 'pool' operation
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))
    # implements eq(1)/(2)


class Cell(nn.Module):
    # FUNCTIONALITY: the cell in Figure 1, has 4 nodes
    # CALL: Network ---> Cell
    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
        # C_prev_prev, C_prev, C are all numbers that specify the sizes (e.g., channel size)
        super(Cell, self).__init__()
        self.reduction = reduction

        if reduction_prev:  # reduction_prev tells whether the previous-previous layer is a 'reduction' layer
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps  # steps = 4
        self._multiplier = multiplier
        # this specifies how many times the number channel is increased, e.g., from C to 4C

        self._ops = nn.ModuleList()  # of size 'step' (4 in this case), stores the nodes
        self._bns = nn.ModuleList()  # not used
        for i in range(self._steps):  # this loop specifies the operations of the 4 nodes in a cell
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                # specifies the stride for one node, reduction means whether this cell is a 'reduction' layer
                op = MixedOp(C, stride)
                self._ops.append(op)
                # in our case, self._ops: [[2 elements], [3 elements], [4 elements], [5 elements]]
                # e.g., for the [4 elements] case, it's the 3rd node and takes input from:
                # 2 previous layers + 1st&2nd nodes

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)  # output of previous-previous layer
        s1 = self.preprocess1(s1)  # output of previous layer

        states = [s0, s1]
        offset = 0
        for i in range(self._steps):
            s = sum(self._ops[offset + j](h, weights[offset + j]) for j, h in enumerate(states))
            # the weights[offset+j] specifies the weights for corresponding operations, in eq(1)/(2),
            # in other words, alpha in eq(3)
            offset += len(states)
            states.append(s)
            # adds the output of current node (which together with s0, s1 will be used as input for next node)

        return torch.cat(states[-self._multiplier:], dim=1)
        # states: [s0, s1, n0, n1, n2, n3]
        # states[-self._multiplier:]: [n0, n1, n2, n3]


class Network(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )
        # deals with the initial input (see the "forward" method of Network)

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
                # the layers at 1/3 and 2/3 of the network are set to be "reduction" layer
            else:
                reduction = False
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr
            # recall that in "forward" method of Cell, its outputs from 4 nodes are concatenated;
            # thus, number of channels are increased, which gives: C_prev = multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            # self.alphas_xxx is of size: (k, num_ops)
            # where: k=sum(1 for i in range(self._steps) for n in range(2 + i))=2+3+4+5
            # take 3 as an example: 3 is the number of outputs from: previous-previous layer, previous-layer, 1st node
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self):
        # FUNCTIONALITY: this initializes the alpha in eq(3)
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self.alphas_reduce = Variable(1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        # FUNCTIONALITY: returns the alpha
        return self._arch_parameters

    def genotype(self):
        # FUNCTIONALITY: gets the optimal architecture (alpha) after optimization

        def _parse(weights):
            # weights: alpha, of shape: (k, num_ops), where k=2+3+4+5
            gene = []
            n = 2
            start = 0
            for i in range(self._steps):
                end = start + n
                W = weights[start:end].copy()  # W of shape: (n, num_ops)
                edges = sorted(range(i + 2),
                               key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
                # i+2 means the previous-previous layer (1), previous-layer (1), and previous nodes (i)
                # sorted(xxx, key=...) means sort xxx based on its key, default in a low-to-high order
                # the key is: the negative of the max value of the operations of W[x],
                # x takes value in (0, 1, ..., i+1), and W[x] =(always) num_ops
                # edges contain top-2 outputs from previous layers/nodes that are with higher weights
                for j in edges:
                    k_best = None
                    for k in range(len(W[j])):
                        if k != PRIMITIVES.index('none'):
                            if k_best is None or W[j][k] > W[j][k_best]:
                                k_best = k
                    gene.append((PRIMITIVES[k_best], j))
                    # sth. like: [('max_pool_3x3', 1), (), ...],
                    # this stores: (what operation to do on an output, which node produces such output)
                start = end
                n += 1
            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        concat = range(2 + self._steps - self._multiplier, self._steps + 2)  # this is: 2, 3, 4, 5
        genotype = Genotype(
            normal=gene_normal, normal_concat=concat,
            reduce=gene_reduce, reduce_concat=concat
        )
        # Genotype is a data structure that specifies the current optimal architecture,
        # some examples are in genotypes.py
        # note that, all the normal/reduction cell use the same kind of architecture (in paper 3.1.1),
        # that's why we only need to focus on the architecture of one cell
        return genotype
