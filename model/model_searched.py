import torch
import torch.nn as nn
from model.operations import *
from torch.autograd import Variable
from guided_filter_pytorch.guided_filter import FastGuidedFilter, GuidedFilter
# ops

Cell_Operation = {
  'Cell_Model': lambda  genotype, C: Cell_Decom(genotype,C),
  'Cell_Fusion': lambda  genotype, C: Cell_Fusion(genotype,C),
  'Cell_Chain': lambda  genotype, C: Cell_Chain(genotype,C),
  'Cell_Chain2': lambda  genotype, C: Cell_Chain2(genotype,C)
}
class MixedOp(nn.Module):

  def __init__(self, C, primitive):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    kernel = 3
    dilation = 1
    if primitive.find('attention') != -1:
        name = primitive.split('_')[0]
        kernel = int(primitive.split('_')[1])
    else:
        name = primitive.split('_')[0]
        kernel = int(primitive.split('_')[1])
        dilation = int(primitive.split('_')[2])
    # print(name, kernel, dilation)
    self._op = OPS[name](C, kernel, dilation, False)

  def forward(self, x):
    return self._op(x)


class Cell_Fusion(nn.Module):

    def __init__(self, genotype, C):
        super(Cell_Fusion, self).__init__()
        # print(C_prev_prev, C_prev, C)
        self.preprocess1 = ReLUConvBN(C, C, 1, 1, 0)
        self.down_2 = DownSample(C, scale_factor=2)
        self.down_4 = DownSample(C, scale_factor=4)
        self.upsample_2 = UpSample(C * 2, scale_factor=2)
        self.upsample_4 = UpSample(C * 4, scale_factor=4)
        self.SKFF = SKFF(C)
        op_names, indices = zip(*genotype.chain_fusion)
        concat = genotype.chain_fusion_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        self._ops_1 = nn.ModuleList()
        self._ops_2 = nn.ModuleList()
        for name, index in zip(op_names, indices):
            print(name, index)
            stride = 1
            op = MixedOp(C, name)
            self._ops += [op]
            op_1 = MixedOp(C * 2, name)
            self._ops_1 += [op_1]
            op_2 = MixedOp(C * 4, name)
            self._ops_2 += [op_2]
        self._indices = indices

    def forward(self, inp_features):
        inp_features = self.preprocess1(inp_features)
        inp_down2 = self.down_2(inp_features)
        inp_down4 = self.down_4(inp_features)
        inputs = [inp_features, inp_down2, inp_down4]
        offset = 0
        states = [inputs]
        for i in range(self._steps):
            s1_0 = self._ops[offset](states[0][0])
            s1_1 = self._ops_1[offset](states[0][1])
            s1_2 = self._ops_2[offset](states[0][2])
            offset += 1
            states = []
            states.append([s1_0, s1_1, s1_2])
        s0 = states[0][0]
        s1 = states[0][1]
        s2 = states[0][2]
        s1_up = self.upsample_2(s1)
        s2_up = self.upsample_4(s2)
        res = self.SKFF([s0, s1_up, s2_up])
        return inp_features + res

    # Cell Chain


class Cell_Chain(nn.Module):

    def __init__(self, genotype, C):
        super(Cell_Chain, self).__init__()
        self.preprocess1 = ReLUConvBN(C, C, 1, 1, 0)
        op_names, indices = zip(*genotype.chain_fusion)
        concat = genotype.chain_fusion_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._steps = len(op_names)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()

        for name, index in zip(op_names, indices):
            print(name, index)
            stride = 1
            op = MixedOp(C, name)
            self._ops += [op]

        self._indices = indices

    def forward(self, inp_features):
        inp_features = self.preprocess1(inp_features)

        inputs = [inp_features]
        offset = 0
        states = [inputs]
        for i in range(self._steps):
            s1_0 = self._ops[offset](states[0][0])

            offset += 1
            states = []
            states.append([s1_0])
        s0 = states[0][0]
        return s0


class Cell_Chain2(nn.Module):

    def __init__(self, genotype, C):
        super(Cell_Chain2, self).__init__()
        self.preprocess1 = ReLUConvBN(C, C, 1, 1, 0)
        op_names, indices = zip(*genotype.normal)
        self._compile(C, op_names, indices)

    def _compile(self, C, op_names, indices):
        assert len(op_names) == len(indices)
        self._steps = len(op_names)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            print(name, index)
            stride = 1
            op = MixedOp(C, name)
            self._ops += [op]
        self._indices = indices

    def forward(self, inp_features):
        inp_features = self.preprocess1(inp_features)

        inputs = [inp_features]
        offset = 0
        states = [inputs]
        for i in range(self._steps):
            s1_0 = self._ops[offset](states[0][0])

            offset += 1
            states = []
            states.append([s1_0])
        s0 = states[0][0]
        return s0

class Cell_Decom(nn.Module):

  def __init__(self, genotype,  C):
    super(Cell_Decom, self).__init__()
    self._C = C
    # self._steps = steps # inner nodes
    self.radiux = [2, 4, 8]
    self.eps_list = [0.001, 0.0001]
    self._ops_1 = nn.ModuleList()
    self._ops_2 = nn.ModuleList()
    op_names, indices = zip(*genotype.chain_model)
    concat = genotype.chain_model_concat
    self._compile(C, op_names, indices, concat)

    self.conv1x1_lf = nn.Conv2d(C*6, C, kernel_size=1, bias=False)
    self.conv1x1_hf = nn.Conv2d(C*6, C, kernel_size=1, bias=False)
    self.conv1x1_concat = nn.Conv2d(C*2, C, kernel_size=1, bias=False)
    self.enchance_concat = EnhanceResidualModule(C)
  def _compile(self, C, op_names, indices, concat):
    assert len(op_names) == len(indices)
    self._steps = len(op_names)
    self._concat = concat
    self.multiplier = len(concat)

    for name, index in zip(op_names, indices):
      print(name,index)
      stride = 1
      op_1 = MixedOp(C,name)
      self._ops_1 += [op_1]
      op_2 =MixedOp(C,name)
      self._ops_2 += [op_2]
    self._indices = indices
  def forward(self, inp_features):
    # inp_reduce = self.conv1x1(inp_features)
    lf, hf = self.decomposition(inp_features,self._C)
    lf = self.conv1x1_lf(lf)
    hf = self.conv1x1_hf(hf)
    offset = 0
    states_lf =[lf]
    for i in range(self._steps):
      s1_0 = self._ops_1[offset](states_lf[0])
      offset += 1
      states_lf = []
      states_lf.append(s1_0)
    # SKFF
    lf = states_lf[0]
    states_hf = [hf]
    offset = 0
    for i in range(self._steps):
      s1_0 = self._ops_2[offset](states_hf[0])
      offset += 1
      states_hf = []
      states_hf.append(s1_0)
    hf = states_hf[0]
    fea_cat = torch.cat([lf,hf],dim=1)
    feature = self.conv1x1_concat(fea_cat)
    feature_res = self.enchance_concat(feature)
    return feature_res

  def get_residue(self, tensor):
    max_channel = torch.max(tensor, dim=1, keepdim=True)
    min_channel = torch.min(tensor, dim=1, keepdim=True)
    res_channel = max_channel[0] - min_channel[0]
    return res_channel

  def decomposition(self, x,C):
    LF_list = []
    HF_list = []
    res = self.get_residue(x)
    res = res.repeat(1, C, 1, 1)
    for radius in self.radiux:
      for eps in self.eps_list:
        self.gf = GuidedFilter(radius, eps)
        LF = self.gf(res, x)
        LF_list.append(LF)
        HF_list.append(x - LF)
    LF = torch.cat(LF_list, dim=1)
    HF = torch.cat(HF_list, dim=1)
    return LF, HF

class ConvLayer(nn.Module):
  def __init__(self, in_dim, out_dim, kernel_size, stride, dilation=1):
      super(ConvLayer, self).__init__()
      reflect_padding = int(dilation * (kernel_size - 1) / 2)
      self.reflection_pad = nn.ReflectionPad2d(reflect_padding)
      self.conv2d = nn.Conv2d(in_dim, out_dim, kernel_size, stride, dilation=dilation)

  def forward(self, x):
      out = self.reflection_pad(x)
      out = self.conv2d(out)
      return out
class Network_Fusion9_Meta(nn.Module):
  def __init__(self, C, layers, genotype, cell_names, multi=False):
      super(Network_Fusion9_Meta, self).__init__()
      self.multi = multi
      self._layers = layers
      self.stem = nn.Sequential(
        ConvLayer(2, C*2, kernel_size=3, stride=1),
        nn.PReLU(),
        ConvLayer(C*2, C, kernel_size=3, stride=1),
        nn.PReLU()
      )
      self.stem_out = nn.Sequential(
          ConvLayer(C, C, kernel_size=3, stride=1),
          nn.PReLU(),
          ConvLayer(C, C // 2, kernel_size=3, stride=1),
          ConvLayer(C // 2, 1, kernel_size=1, stride=1)

      )

      self.cells = nn.ModuleList()
      for i in range(self._layers+1):
        op = Cell_Operation[cell_names[i]](genotype, C)
        self.cells += [op]

  def forward(self, lrr, vis):
      input_new = torch.cat([lrr,vis],dim=1)
      s1 = self.stem(input_new)
      for i in range(self._layers+1):
          s1 = self.cells[i](s1)
      fused = self.stem_out(s1) # the first result

      return  fused




class Network_Fusion9_2(nn.Module):
  def __init__(self, C, layers, cell_names2,genotypes, multi=False):
      super(Network_Fusion9_2, self).__init__()
      self.multi = multi
      self._layers = layers
      self.stem2 = nn.Sequential(
          ConvLayer(2, C * 2, kernel_size=3, stride=1),
          # nn.BatchNorm2d(C),
          nn.PReLU(),
          ConvLayer(C * 2, C, kernel_size=3, stride=1),
          nn.BatchNorm2d(C),
          nn.PReLU()
      )
      self.stem_out2 = nn.Sequential(
          ConvLayer(C, C, kernel_size=3, stride=1),
          # nn.BatchNorm2d(C),
          nn.PReLU(),
          ConvLayer(C, C // 2, kernel_size=3, stride=1),
          ConvLayer(C // 2, 1, kernel_size=1, stride=1)

      )
      self.stem3 = nn.Sequential(
          ConvLayer(2, C * 2, kernel_size=3, stride=1),
          # nn.BatchNorm2d(C),
          nn.PReLU(),
          ConvLayer(C * 2, C, kernel_size=3, stride=1),
          nn.BatchNorm2d(C),
          nn.PReLU()
      )
      self.stem_out3 = nn.Sequential(
          ConvLayer(C, C, kernel_size=3, stride=1),
          # nn.BatchNorm2d(C),
          nn.PReLU(),
          ConvLayer(C, C // 2, kernel_size=3, stride=1),
          ConvLayer(C // 2, 1, kernel_size=1, stride=1)

      )

      self.stem_out4 = nn.Sequential(
          ConvLayer(C*2, C, kernel_size=3, stride=1),
          # nn.BatchNorm2d(C),
          ConvLayer(C, C // 2, kernel_size=3, stride=1),
          ConvLayer(C // 2, 1, kernel_size=1, stride=1)

      )
      self.tanh = nn.Tanh()
      self.sig = nn.Sigmoid()
      self.cells = nn.ModuleList()
      self.cells2 = nn.ModuleList()
      for i in range(self._layers+1):
        op = Cell_Operation[cell_names2[0]](genotypes[0], C)
        self.cells2 += [op]
      self.cells3 = nn.ModuleList()
      for i in range(self._layers+1):
        op = Cell_Operation[cell_names2[0]](genotypes[1], C)
        self.cells3 += [op]

  def forward(self, lrr, vis, fused):
      fused_irr = self.stem2(torch.cat([lrr,fused],dim=1))
      fused_vis = self.stem3(torch.cat([vis,fused],dim=1))
      for i in range(self._layers+1):
          fused_vis = self.cells2[i](fused_vis)
      for i in range(self._layers+1):
          fused_irr = self.cells3[i](fused_irr)
      fused_lrr_img = self.tanh(self.stem_out2(fused_irr))
      fused_vis_img = self.tanh(self.stem_out3(fused_vis))
      final_img = self.tanh(self.stem_out4(torch.cat((fused_irr ,fused_vis),dim=1)))
      return fused_lrr_img,fused_vis_img,final_img


class Network_Fusion9_3(nn.Module):
  # using the binary map
  def __init__(self, C, layers, cell_names2,genotypes, multi=False):
      super(Network_Fusion9_3, self).__init__()
      self.multi = multi
      self._layers = layers
      self.spa = spatial_attn_layer2()
      # the final output
      self.stem2 = nn.Sequential(
          ConvLayer(2, C * 2, kernel_size=3, stride=1),
          # nn.BatchNorm2d(C),
          nn.PReLU(),
          ConvLayer(C * 2, C, kernel_size=3, stride=1),
          nn.BatchNorm2d(C),
          nn.PReLU()
      )
      self.stem_out2 = nn.Sequential(
          ConvLayer(C, C, kernel_size=3, stride=1),
          # nn.BatchNorm2d(C),
          nn.PReLU(),
          ConvLayer(C, C // 2, kernel_size=3, stride=1),
          ConvLayer(C // 2, 1, kernel_size=1, stride=1)

      )
      self.stem3 = nn.Sequential(
          ConvLayer(2, C * 2, kernel_size=3, stride=1),
          # nn.BatchNorm2d(C),
          nn.PReLU(),
          ConvLayer(C * 2, C, kernel_size=3, stride=1),
          nn.BatchNorm2d(C),
          nn.PReLU()
      )
      self.stem_out3 = nn.Sequential(
          ConvLayer(C, C, kernel_size=3, stride=1),
          # nn.BatchNorm2d(C),
          nn.PReLU(),
          ConvLayer(C, C // 2, kernel_size=3, stride=1),
          ConvLayer(C // 2, 1, kernel_size=1, stride=1)

      )

      self.stem_out4 = nn.Sequential(
          ConvLayer(C, C, kernel_size=3, stride=1),
          # nn.BatchNorm2d(C),
          ConvLayer(C, C // 2, kernel_size=3, stride=1),
          ConvLayer(C // 2, 1, kernel_size=1, stride=1)

      )
      self.tanh = nn.Tanh()
      self.sig = nn.Sigmoid()
      self.cells = nn.ModuleList()
      self.cells2 = nn.ModuleList()
      for i in range(self._layers+1):
        op = Cell_Operation[cell_names2[0]](genotypes[0], C)
        self.cells2 += [op]
      self.cells3 = nn.ModuleList()
      for i in range(self._layers+1):
        op = Cell_Operation[cell_names2[0]](genotypes[1], C)
        self.cells3 += [op]

  def forward(self, lrr, vis, fused):
      fused_irr = self.stem2(torch.cat([lrr,fused],dim=1))
      fused_vis = self.stem3(torch.cat([vis,fused],dim=1))
      for i in range(self._layers+1):
          fused_vis = self.cells2[i](fused_vis)
      for i in range(self._layers+1):
          fused_irr = self.cells3[i](fused_irr)
      fused_lrr_img = self.tanh(self.stem_out2(fused_irr))
      fused_vis_img = self.tanh(self.stem_out3(fused_vis))
      mask = self.spa(fused_irr)
      final_img = (self.stem_out4(mask*fused_irr + (1-mask)*fused_vis))
      return fused_lrr_img,fused_vis_img,final_img