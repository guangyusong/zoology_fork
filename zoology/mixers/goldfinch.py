import torch
from torch import nn, Tensor
import torch.nn.functional as F
from zoology.mixers.rwkv_goldfinch.CoreDependencies import *
from zoology.mixers.rwkv_goldfinch.cuda6 import RUN_CUDA_RWKV6

from zoology.mixers.rwkv_goldfinch.tmix import TimeMixState

import math

from typing import Tuple

from zoology.mixers.rwkv_goldfinch.rotary import generate_rotary_embedding, generate_binary_rotary_embedding, apply_rotary_embedding
from zoology.mixers.rwkv_goldfinch.norm import rms_norm

class GPTAlpha_Tmix_goco(MyModule):
    def __init__(self, l_max: int, d_model, layer_idx, angles=None, bias_mask=None, n_layer = 12):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_model = d_model
        self.n_layer = n_layer
        self.dim_att = d_model
        self.ctx_len = l_max
        
        os.environ['RWKV_CTXLEN'] = str(l_max)
        self.head_size_a = 64

        self.k_head_size = self.v_head_size = self.head_size = self.head_size_a
        self.n_kv_head = self.n_head = self.dim_att // self.head_size
        assert self.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_idx / (self.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_idx / self.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, self.d_model)
            for i in range(self.d_model):
                ddd[0, 0, i] = i / self.d_model

            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_q = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            self.time_maa_v_cache = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            D_MIX_LORA = 32
            self.time_maa_q_w1 = nn.Parameter(torch.zeros(self.d_model, D_MIX_LORA))
            self.time_maa_q_w2 = nn.Parameter(torch.empty(D_MIX_LORA, self.d_model).uniform_(-0.01, 0.01))
            self.time_maa_kv_w1 = nn.Parameter(torch.zeros(self.d_model, D_MIX_LORA*2))
            self.time_maa_kv_w2 = nn.Parameter(torch.empty(2, D_MIX_LORA, self.d_model).uniform_(-0.01, 0.01))

            D_VALUE_LORA = max(self.d_model // 16, 64)
            self.time_key_w1 = nn.Parameter(torch.zeros(self.d_model, D_VALUE_LORA))
            self.time_key_w2 = nn.Parameter(torch.zeros(D_VALUE_LORA, self.dim_att).uniform_(-0.01, 0.01))
            self.time_value_w1 = nn.Parameter(torch.zeros(self.d_model, D_VALUE_LORA))
            self.time_value_w2 = nn.Parameter(torch.zeros(D_VALUE_LORA, self.dim_att).uniform_(-0.01, 0.01))

        self.query = nn.Linear(self.d_model, self.dim_att, bias=False)
        self.output = nn.Linear(self.dim_att, self.d_model, bias=False)
        self.ln_q = nn.LayerNorm(self.dim_att)
        self.ln_k = nn.LayerNorm(self.dim_att)
        self.ln_v = nn.LayerNorm(self.dim_att)
        self.ln_x = nn.LayerNorm(self.dim_att)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        self.angles = angles
        self.bias_mask = bias_mask

    def shift_cat(self, x):
        return torch.cat([x[:, :1], x[:, :-1]], dim=1)


    @MyFunction
    def forward(self, x, xo, k_cache, last_time_mix_state:TimeMixState):
        B, T, C = x.size()
        H = self.n_head
        K = C // H
        V = C // H

        shift_state = x[:, -1].clone()
        dxprev = torch.concat((last_time_mix_state.shift_state.unsqueeze(1), x[:, :-1]), dim=1) - x

        xxx = x + dxprev * self.time_maa_x
        mq = torch.tanh(xxx @ self.time_maa_q_w1) @ self.time_maa_q_w2

        xo = rms_norm(xo)
        dxo_prev = self.time_shift(xo) - xo
        xxx = xo + dxo_prev * self.time_maa_v_cache
        xxx = torch.tanh(xxx @ self.time_maa_kv_w1).view(B*xo.size(1), self.time_maa_kv_w2.size(0), -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_kv_w2).view(self.time_maa_kv_w2.size(0), B, xo.size(1), C)
        mk, mv = xxx.unbind(dim=0)

        k = k_cache
        dkprev = self.time_shift(k) - k
        v = xo
        dvprev = self.time_shift(v) - v      

        xq = x + dxprev * (self.time_maa_q + mq)
        k = k + dkprev * (self.time_maa_k + mk)
        v = v + dvprev * (self.time_maa_v + mv)
        
        k = k + torch.tanh(k @ self.time_key_w1) @ self.time_key_w2
        v = v + torch.tanh(v @ self.time_value_w1) @ self.time_value_w2
       
        q = self.query(xq)
        
        q = self.ln_q(q)
        k = self.ln_k(k)
        v = self.ln_v(v)

        q = q.view(B,-1,H,K).transpose(1,2)
        k = k.view(B,-1,H,K).transpose(1,2)
        v = v.view(B,-1,H,V).transpose(1,2)
        
        if self.angles is not None:
            self.angles = self.angles.to(x.device)
            q, k = apply_rotary_embedding(q, k, self.angles)


        # causality MUST be enforced for longer runs because even though we won't use the results at t-1 the next chanmix WILL for its tokenshift!
        # this is also why we must allow through the last MANY time-steps if we have that many, so chanmix receives both of these and can lerp between those results!
        # the results can tokenshift their way forward up to one full timestep each layer via chanmix, so we really have to keep up to all N goldfinch layers around

        x = nn.functional.scaled_dot_product_attention(q,k,v,is_causal=q.size(-2)>1)

        x = x.transpose(1,2).reshape(B,-1,C)
       
        x = self.ln_x(x)
        #x = F.layer_norm(x.float(), self.ln_x.normalized_shape, self.ln_x.weight.float(), self.ln_x.bias.float()).to(x.dtype)

        x = self.output(x)

        return x, TimeMixState(last_time_mix_state.wkv_state, shift_state)
