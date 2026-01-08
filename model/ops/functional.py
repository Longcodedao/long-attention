# holo_gpt/ops/functional.py
import torch
import triton
from .kernels import _holo_scan_fwd_kernel, _holo_scan_bwd_kernel

class HoloScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, k, q):
        # 1. Contiguity & Save
        if v.stride(-1) != 1: v = v.contiguous()
        if k.stride(-1) != 1: k = k.contiguous()
        if q.stride(-1) != 1: q = q.contiguous()
        
        ctx.save_for_backward(v, k, q)
        
        # 2. Setup
        B, T, H, D = v.shape
        out = torch.empty((B, T, H, D), device=v.device, dtype=torch.float32)
        
        # 3. Views & Strides
        v_f, k_f, q_f = v.view(torch.float32), k.view(torch.float32), q.view(torch.float32)
        
        def get_strides(x):
            return (x.stride(0)*2, x.stride(1)*2, x.stride(2)*2, x.stride(3)*2)

        grid = (B * H, )
        BLOCK_D = triton.next_power_of_2(D)
        
        # 4. Launch
        _holo_scan_fwd_kernel[grid](
            v_f, k_f, q_f, out,
            *get_strides(v), *get_strides(k), *get_strides(q), *out.stride(),
            T=T, H=H, D=D, BLOCK_D=BLOCK_D
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        v, k, q = ctx.saved_tensors
        B, T, H, D = v.shape
        grad_output = grad_output.contiguous()
        
        dv, dk, dq = torch.empty_like(v), torch.empty_like(k), torch.empty_like(q)
        
        # Float Views
        v_f, k_f, q_f = v.view(torch.float32), k.view(torch.float32), q.view(torch.float32)
        dv_f, dk_f, dq_f = dv.view(torch.float32), dk.view(torch.float32), dq.view(torch.float32)
        
        def get_strides(x):
            return (x.stride(0)*2, x.stride(1)*2, x.stride(2)*2, x.stride(3)*2)

        grid = (B * H, )
        BLOCK_D = triton.next_power_of_2(D)
        
        _holo_scan_bwd_kernel[grid](
            v_f, k_f, q_f, grad_output,
            dv_f, dk_f, dq_f,
            *get_strides(v), *get_strides(k), *get_strides(q), *grad_output.stride(),
            *get_strides(dv), *get_strides(dk), *get_strides(dq),
            T=T, H=H, D=D, BLOCK_D=BLOCK_D
        )
        return dv, dk, dq


# The public API
def holo_scan(v, k, q):
    return HoloScanFunction.apply(v, k, q)