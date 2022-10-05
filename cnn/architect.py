import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])
    # this flattens a set of parameters to one dimensional vector


class Architect(object):
    # this is the wrapper for model, which gives the model some extensions, its function 'step' is used in
    # the "train" function of train_search.py, which does the bi-level optimization process

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        # the model has two sets of parameters:
        # 1. arch_parameters() [the outer layer variable alpha]    2. parameters() [the inner layer variable w]
        # here only the alpha will be updated

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        # FUNCTIONALITY:
        # this function is used to compute the model, where w* is replaced with "w-eta·Grad_w L_train" in eq(6)
        # CALL:
        # step() ---> _backward_step_unrolled() ---> compute_unrolled_model()
        # INPUT:
        # input, target: training set data and ground truth
        # eta: learning rate
        # network optimizer:  SGD optimizer (passed by "train" function in train_search.py)
        loss = self.model._loss(input, target)
        theta = _concat(self.model.parameters()).data
        # theta here is the "w", which is flattened to one dimensional vectors

        # the following code is the SGD algorithm,
        # see details here: https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(
                self.network_momentum)
            # calculates the momentum for SGD algorithm
        except:
            moment = torch.zeros_like(theta)
            # no momentum (momentum=0)
        dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay * theta
        # the "_concat(...)" is the gradient of "w" (self.model.parameters()), and the rest is the weight-decay term
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment + dtheta))
        # the input of "_construct_model_from_theta()" is: theta - eta·(Grad(theta)L_train + momentum), which is the
        # approximated one-step gradient descent "w-eta·Grad(w)L_train"
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
            # does not replace w with "w-eta·Grad(w)L_train" in eq(6)
        self.optimizer.step()
        # updates the model's alpha and move forward

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        # unrolled_model is a model whose "w" is changed to "w-eta·Grad(w)L_train", as shown in eq(6)
        unrolled_loss = unrolled_model._loss(input_valid, target_valid)
        # this is L_val(w-eta·Grad(w)L_train, alpha) in eq(6)

        unrolled_loss.backward()  # calculated the gradient
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        # the gradient for alpha in eq(7)
        vector = [v.grad.data for v in unrolled_model.parameters()]
        # "Grad(w)L_val(w-eta·Grad(w)L_train, alpha)", in eq(7)
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        # computes second term "Hessian·Grad(w)L_val" in eq(7)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)
            # calculate eq(7)

        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)
            # updates the gradient of alpha

    def _construct_model_from_theta(self, theta):
        # FUNCTIONALITY:
        # this function is used to construct a new model, where only its "w" is updated
        # CALL:
        # step() ---> _backward_step_unrolled() ---> compute_unrolled_model() ---> _construct_model_from_theta
        # INPUT:
        # theta: "w-eta·Grad(w)L_train" in eq(6)
        model_new = self.model.new()  # initialize a new model
        model_dict = self.model.state_dict()  # get the states of the "old" model (whose "w" is not updated)

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            # v.size() is the shape of the tensor, e.g.,: (3, 4), and np.prod(): 3*4--->12
            params[k] = theta[offset: offset + v_length].view(v.size())
            # prepares the "new" parameters for model_new
            # theta is a one-dimensional vector, where trunk [offset: offset + v_length] stores the weights of
            # a component for the new model
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        # FUNCTIONALITY:
        # this function is used to compute the "Hessian·Grad(w)L_val" in eq(8)
        # CALL:
        # step() ---> _backward_step_unrolled() ---> _hessian_vector_product()
        # INPUT:
        # vector: "Grad(w)L_val(w-eta·Grad(w)L_train, alpha)", in eq(7)
        # input, target: training set data and ground truth
        R = r / _concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
            # w+ in eq(8)
        loss = self.model._loss(input, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())
        # Grad(alpha)L_train(w+, alpha)

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(2 * R, v)
            # w- in eq(8)  (from w+ to w-, that's why its 2*R)
        loss = self.model._loss(input, target)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())
        # Grad(alpha)L_train(w-, alpha)

        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(R, v)
            # change w- to w, restore the weights as how it is in the beginning of this function

        return [(x - y).div_(2 * R) for x, y in zip(grads_p, grads_n)]
