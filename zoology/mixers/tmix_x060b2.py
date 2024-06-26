import torch
from torch import nn, Tensor
import torch.nn.functional as F
from zoology.mixers.rwkv_goldfinch.CoreDependencies import *
from zoology.mixers.rwkv_goldfinch.cuda6 import RUN_CUDA_RWKV6

from zoology.mixers.rwkv_goldfinch.tmix import TimeMixState

class RWKV_Tmix_x060b2(MyModule):
    def __init__(self, l_max: int, d_model, layer_idx, n_layer = 12):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_embd = d_model
        self.dim_att = d_model
        self.head_size = 64
        self.n_layer = n_layer
        
        self.n_head = self.dim_att // self.head_size
        assert self.dim_att % self.n_head == 0

        self.use_one_minus_w = 1 # use_one_minus_w
        self.use_v2 = 1 # use_v2

        with torch.no_grad():
            ratio_0_to_1 = layer_idx / (self.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_idx / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.n_embd)
            for i in range(self.n_embd):
                ddd[0, 0, i] = i / self.n_embd

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            # self.time_maa_all = nn.Parameter(torch.cat([
            #     1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0), # r
            #     1.0 - torch.pow(ddd, ratio_1_to_almost0), # k
            #     1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1), # v
            #     1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1), # v2
            #     1.0 - torch.pow(ddd, ratio_1_to_almost0), # w
            # ]))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v2 = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            D_MIX_LORA = 32
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, self.n_embd).uniform_(-0.01, 0.01))
            self.time_maa_w1 = nn.Parameter(torch.zeros(self.n_embd, D_MIX_LORA*self.time_maa_w2.size(0)))

            decay_speed = torch.ones(self.dim_att)
            for n in range(self.dim_att):
                decay_speed[n] = -6 + 5 * (n / (self.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,self.dim_att))
            D_DECAY_LORA = 64
            self.time_decay_w1 = nn.Parameter(torch.zeros(self.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, self.dim_att).uniform_(-0.01, 0.01))

            self.time_value2_w1 = nn.Parameter(torch.zeros(self.n_embd, D_DECAY_LORA))
            self.time_value2_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, self.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(self.dim_att)
            for n in range(self.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (self.dim_att - 1))) + zigzag
            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.receptance = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.key = nn.Linear(self.n_embd, self.dim_att, bias=False)

        self.value = nn.Linear(self.n_embd, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, self.n_embd, bias=False)
        self.ln_x = nn.LayerNorm(self.dim_att)

    @MyFunction
    def forward(self, x, xo, kv_cache, last_state:TimeMixState):
        B, T, C = x.size()
        H = self.n_head

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, self.time_maa_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(self.time_maa_w2.size(0), B, T, C)

        # xr, xk, xv, xv2, xw = (x + dxprev * (self.time_maa_all.view(5, 1, 1, C) + xxx)).unbind(dim=0)
        mr, mk, mv, mw, mv2 = xxx.unbind(dim=0)
        xr = x + dxprev * (self.time_maa_r + mr)
        xk = x + dxprev * (self.time_maa_k + mk)
        xv = x + dxprev * (self.time_maa_v + mv)
        xw = x + dxprev * (self.time_maa_w + mw)
        xv2 = x + dxprev * (self.time_maa_v2 + mv2)
        
        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        v2 = self.value(xv2) + torch.tanh(xv2 @ self.time_value2_w1) @ self.time_value2_w2
        w = self.time_decay + torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2

        if self.use_one_minus_w:
            k = k * (1 - (-w.exp()).exp())

        if self.use_v2:
            u = torch.zeros_like(self.time_faaaa)
        else:
            u = self.time_faaaa

        wkv_state = last_state.wkv_state.clone()
        y = RUN_CUDA_RWKV6(B, T, C, H, r, k, v, w, u, wkv_state)

        if self.use_v2:
            y = y + v2

        y = self.ln_x(y)
        #y = F.layer_norm(y.float(), self.ln_x.normalized_shape, self.ln_x.weight.float(), self.ln_x.bias.float()).to(y.dtype)

        y = self.output(y)
        return y, TimeMixState(wkv_state, shift_state)