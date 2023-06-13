import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat2(xs,moment):
  xs_new = []
  for m,x in zip(moment,xs):
    if x is not None:
      xs_new.append(x.view(-1))
    else:
      xs_new.append(torch.zeros_like(m).view(-1))
  return torch.cat(xs_new)

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])

class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, lamda,latency,input, target, eta, network_optimizer):
    loss = self.model._loss( input, target,lamda,latency)
    theta = _concat(self.model.parameters()).data

    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    # print('models...............')
    # print(self.model)
    # print(theta.shape)
    # Why
    dtheta = _concat2(torch.autograd.grad(loss, self.model.parameters(),allow_unused=True),self.model.parameters()).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, lamda, latency, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(lamda,latency,input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid,lamda,latency)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid,lamda,latency):
    loss = self.model._loss(input_valid, target_valid,lamda,latency)
    loss.backward()

  def _backward_step_unrolled(self,lamda,latency, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    # unrolled_model = self._compute_unrolled_model(lamda,latency,input_train, target_train, eta, network_optimizer)
    # unrolled_loss = unrolled_model._loss(input_valid, target_valid,lamda,latency)
    #

    # # modify


    ##
    # unrolled_model = self._compute_unrolled_model(lamda, latency, input_train, target_train, eta, network_optimizer)
    unrolled_loss = self.model._loss(input_valid, target_valid, lamda, latency)
    # upper_loss
    # lower_loss
    unrolled_loss.backward()
    dalpha = [v.grad for v in self.model.arch_parameters()]
    vector = []
    for v in self.model.parameters():
      if v.grad is not None:
        vector.append(v.grad.data)
      else:
        vector.append(torch.zeros_like(v))

    # lower_loss = self.model._loss(input_train, target_train, lamda, latency)
    lower_loss_ = self.model._loss(input_train, target_train, lamda, latency)

    # dFy = torch.autograd.grad(upper_loss, unrolled_model.parameters(),allow_unused=True)

    dfy = torch.autograd.grad(lower_loss_, self.model.parameters(), allow_unused=True)
    gfyfy = 0
    gFyfy = 0
    for f, F in zip(dfy, vector):
      if f is  None:
        f = torch.zeros_like(F)
      gfyfy = gfyfy + torch.sum(f * f)
      gFyfy = gFyfy + torch.sum(F * f)

    lower_loss_2 = self.model._loss(input_train, target_train, lamda, latency)
    GN_loss = -gFyfy.detach() / gfyfy.detach() * lower_loss_2
    implicit_grads = torch.autograd.grad(GN_loss, self.model.arch_parameters(), allow_unused=True)

    # vector = [v.grad.data for v in unrolled_model.parameters()]
    # implicit_grads = self._hessian_vector_product(vector, input_train, target_train,)

    for g, ig in zip(dalpha, implicit_grads):
      if ig is None:
        ig = torch.zeros_like(g)
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target,0,100)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target,0,100)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

