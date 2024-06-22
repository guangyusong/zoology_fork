import os
import torch
from torch import nn, Tensor
from .CoreDependencies import *

########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

os.environ["RWKV_HEAD_SIZE_A"] = "64"
os.environ["RWKV_CTXLEN"] = "512"
os.environ["RWKV_MODEL_TYPE"] = "x060"

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])

if 'x060' in os.environ["RWKV_MODEL_TYPE"]:
    extra_cuda_cflags = ["-O3", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"]
    if torch.cuda.is_available():
        if torch.version.hip:
            extra_cuda_cflags += ["--save-temps"]
        else:
            extra_cuda_cflags += ["-res-usage", "--use_fast_math", "-Xptxas -O3", "--extra-device-vectorization"]

    # Calculate absolute paths to the source files
    script_dir = os.path.dirname(os.path.realpath(__file__))
    cpp_file = os.path.join(script_dir, "wkv6_op.cpp")
    cu_file = os.path.join(script_dir, "wkv6_cuda.cu")
    wkv6_cuda = load(name="wkv6", sources=[cpp_file, cu_file],
                        verbose=True, extra_cuda_cflags=extra_cuda_cflags)
            
    class WKV_6(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u, s):
            with torch.no_grad():
                dtype = r.dtype
                assert r.dtype == dtype
                assert k.dtype == dtype
                assert v.dtype == dtype
                assert w.dtype == dtype
                assert u.dtype == dtype
                assert s.dtype == dtype
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                ctx.dtype = dtype
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                assert s.is_contiguous()
                ctx.save_for_backward(r, k, v, w, u)
                y = torch.empty((B, T, C), device=r.device, dtype=dtype, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                if dtype == torch.bfloat16:
                    wkv6_cuda.forward_bf16(B, T, C, H, r, k, v, w, u, s, y)
                elif dtype == torch.float16:
                    wkv6_cuda.forward_fp16(B, T, C, H, r, k, v, w, u, s, y)
                elif dtype == torch.float32:
                    wkv6_cuda.forward_fp32(B, T, C, H, r, k, v, w, u, s, y)
                else:
                    raise ValueError(f"Unsupported dtype {dtype} for WKV_6")
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                dtype = ctx.dtype
                gy = gy.float()
                assert gy.dtype == dtype
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                if not gy.is_contiguous():
                    gy = gy.contiguous()
                assert gy.is_contiguous()
                r, k, v, w, u = ctx.saved_tensors
                r, k, v, w, u = r.float(), k.float(), v.float(), w.float(), u.float()
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                #gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format)#.uniform_(-100, 100)
                gr, gk, gv, gw, gu = gr.float(), gk.float(), gv.float(), gw.float(), gu.float()
                if dtype == torch.bfloat16:
                    wkv6_cuda.backward_bf16(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                elif dtype == torch.float16:
                    wkv6_cuda.backward_fp16(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                elif dtype == torch.float32:
                    wkv6_cuda.backward_fp32(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu)
                else:
                    raise ValueError(f"Unsupported dtype {dtype} for WKV6_CUDA")
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu, None)

    @TCompileDisable 
    @torch.jit.ignore
    def RUN_CUDA_RWKV6(B:int, T:int, C:int, H:int, r, k, v, w, u, s):
        return WKV_6.apply(B, T, C, H, r, k, v, w, u, s)
else:
    @TCompileDisable 
    @torch.jit.ignore
    def RUN_CUDA_RWKV6(B:int, T:int, C:int, H:int, r, k, v, w, u, s):
        return None    