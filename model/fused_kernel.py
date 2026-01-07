import torch 
import triton 
import triton.language as tl 


# Forward kernel
@triton.jit
def _holo_scan_fwd_kernel(
    v_ptr, k_ptr, q_ptr, out_ptr,
    # Strides (Input V, K, Q are complex, so strides are doubled for float pointer math)
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_q_b, stride_q_t, stride_q_h, stride_q_d,
    stride_out_b, stride_out_t, stride_out_h, stride_out_d,
    T, H, D, 
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    b_idx = pid // H
    h_idx = pid % H
    
    # Base pointers
    v_base = v_ptr + b_idx * stride_v_b + h_idx * stride_v_h
    k_base = k_ptr + b_idx * stride_k_b + h_idx * stride_k_h
    q_base = q_ptr + b_idx * stride_q_b + h_idx * stride_q_h
    out_base = out_ptr + b_idx * stride_out_b + h_idx * stride_out_h

    # Offsets
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    
    # Accumulator
    acc_real = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_imag = tl.zeros([BLOCK_D], dtype=tl.float32)

    for t in range(T):
        off_v = t * stride_v_t + offs_d * stride_v_d
        off_k = t * stride_k_t + offs_d * stride_k_d
        off_q = t * stride_q_t + offs_d * stride_q_d
        
        # Load V, K, Q
        # Note: stride_d is 2 floats (Real, Imag). 
        # Base + offset points to Real. Base + offset + 1 points to Imag.
        vr = tl.load(v_base + off_v, mask=mask_d, other=0.0)
        vi = tl.load(v_base + off_v + 1, mask=mask_d, other=0.0)
        kr = tl.load(k_base + off_k, mask=mask_d, other=0.0)
        ki = tl.load(k_base + off_k + 1, mask=mask_d, other=0.0)
        qr = tl.load(q_base + off_q, mask=mask_d, other=0.0)
        qi = tl.load(q_base + off_q + 1, mask=mask_d, other=0.0)

        # Acc += V * K
        term_r = vr * kr - vi * ki
        term_i = vr * ki + vi * kr
        acc_real += term_r
        acc_imag += term_i
        
        # Out = Real(Acc * Q) / scale
        # (Ar + jAi) * (Qr + jQi) = (ArQr - AiQi) + j(...)
        out_val = acc_real * qr - acc_imag * qi
        
        scale = tl.sqrt((t + 1).to(tl.float32))
        out_val = out_val / scale

        off_out = t * stride_out_t + offs_d * stride_out_d
        tl.store(out_base + off_out, out_val, mask=mask_d)


# Backward kernel
@triton.jit
def _holo_scan_bwd_kernel(
    # Pointers
    v_ptr, k_ptr, q_ptr, do_ptr, # Inputs
    dv_ptr, dk_ptr, dq_ptr,      # Outputs
    # Strides (Input V, K, Q and Output dV, dK, dQ are doubled for Float view of Complex)
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_q_b, stride_q_t, stride_q_h, stride_q_d,
    stride_do_b, stride_do_t, stride_do_h, stride_do_d, # d_out is float, normal strides
    stride_dv_b, stride_dv_t, stride_dv_h, stride_dv_d,
    stride_dk_b, stride_dk_t, stride_dk_h, stride_dk_d,
    stride_dq_b, stride_dq_t, stride_dq_h, stride_dq_d,
    # Constants
    T, H, D,
    BLOCK_D: tl.constexpr
):
    pid = tl.program_id(0)
    b_idx = pid // H
    h_idx = pid % H

    # 1. Setup Base Pointers
    # Inputs
    v_base = v_ptr + b_idx * stride_v_b + h_idx * stride_v_h
    k_base = k_ptr + b_idx * stride_k_b + h_idx * stride_k_h
    q_base = q_ptr + b_idx * stride_q_b + h_idx * stride_q_h
    do_base = do_ptr + b_idx * stride_do_b + h_idx * stride_do_h
    
    # Outputs
    dv_base = dv_ptr + b_idx * stride_dv_b + h_idx * stride_dv_h
    dk_base = dk_ptr + b_idx * stride_dk_b + h_idx * stride_dk_h
    dq_base = dq_ptr + b_idx * stride_dq_b + h_idx * stride_dq_h

    # Offsets
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # =========================================================
    # Phase 1: Forward Re-computation of Accumulator (Acc)
    # =========================================================
    # We need the state of Acc at every step to compute dQ.
    # Instead of storing it, we compute the FINAL state first,
    # then backtrack during the backward loop.
    
    acc_r = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc_i = tl.zeros([BLOCK_D], dtype=tl.float32)

    for t in range(T):
        off_v = t * stride_v_t + offs_d * stride_v_d
        off_k = t * stride_k_t + offs_d * stride_k_d
        
        vr = tl.load(v_base + off_v, mask=mask_d, other=0.0)
        vi = tl.load(v_base + off_v + 1, mask=mask_d, other=0.0)
        kr = tl.load(k_base + off_k, mask=mask_d, other=0.0)
        ki = tl.load(k_base + off_k + 1, mask=mask_d, other=0.0)

        # Acc += V * K
        acc_r += vr * kr - vi * ki
        acc_i += vr * ki + vi * kr

    # =========================================================
    # Phase 2: Backward Gradient Calculation
    # =========================================================
    
    # Gradient Accumulator for the Scan operation (Reverse Scan)
    # This represents dL/d(Acc)
    d_acc_r = tl.zeros([BLOCK_D], dtype=tl.float32)
    d_acc_i = tl.zeros([BLOCK_D], dtype=tl.float32)

    # Iterate T backwards: T-1 ... 0
    for t in range(T - 1, -1, -1):
        # -------------------------------------------------
        # A. Load Inputs for this timestep
        # -------------------------------------------------
        off_v = t * stride_v_t + offs_d * stride_v_d
        off_k = t * stride_k_t + offs_d * stride_k_d
        off_q = t * stride_q_t + offs_d * stride_q_d
        off_do = t * stride_do_t + offs_d * stride_do_d

        vr = tl.load(v_base + off_v, mask=mask_d, other=0.0)
        vi = tl.load(v_base + off_v + 1, mask=mask_d, other=0.0)
        kr = tl.load(k_base + off_k, mask=mask_d, other=0.0)
        ki = tl.load(k_base + off_k + 1, mask=mask_d, other=0.0)
        qr = tl.load(q_base + off_q, mask=mask_d, other=0.0)
        qi = tl.load(q_base + off_q + 1, mask=mask_d, other=0.0)
        
        # d_out is Real
        dout = tl.load(do_base + off_do, mask=mask_d, other=0.0)
        
        # Scale
        scale = tl.sqrt((t + 1).to(tl.float32))
        inv_scale = 1.0 / scale
        
        # -------------------------------------------------
        # B. Compute Gradient w.r.t Q (dQ)
        # -------------------------------------------------
        # Forward: Out = Re(Acc * Q) / scale
        # dQ = dOut * (1/scale) * conj(Acc)
        
        term_dout = dout * inv_scale
        
        # dQ = term * conj(Acc) -> (Acc_r - j Acc_i)
        dqr = term_dout * acc_r
        dqi = term_dout * (-acc_i)
        
        off_dq = t * stride_dq_t + offs_d * stride_dq_d
        tl.store(dq_base + off_dq, dqr, mask=mask_d)
        tl.store(dq_base + off_dq + 1, dqi, mask=mask_d)

        # -------------------------------------------------
        # C. Update Accumulator Gradient (Reverse Scan)
        # -------------------------------------------------
        # The gradient flows from Out -> Acc
        # dAcc_local = dOut * (1/scale) * conj(Q)
        
        d_acc_local_r = term_dout * qr
        d_acc_local_i = term_dout * (-qi)
        
        # Add to running sum
        d_acc_r += d_acc_local_r
        d_acc_i += d_acc_local_i
        
        # -------------------------------------------------
        # D. Compute Gradients w.r.t V and K (dV, dK)
        # -------------------------------------------------
        # P = V * K.  dL/dP = dAcc (running sum)
        # dV = dP * conj(K)
        # dK = dP * conj(V)
        
        # dV = (d_acc_r + j d_acc_i) * (kr - j ki)
        dvr = d_acc_r * kr - d_acc_i * (-ki)
        dvi = d_acc_r * (-ki) + d_acc_i * kr
        
        # dK = (d_acc_r + j d_acc_i) * (vr - j vi)
        dkr = d_acc_r * vr - d_acc_i * (-vi)
        dki = d_acc_r * (-vi) + d_acc_i * vr
        
        off_dv = t * stride_dv_t + offs_d * stride_dv_d
        tl.store(dv_base + off_dv, dvr, mask=mask_d)
        tl.store(dv_base + off_dv + 1, dvi, mask=mask_d)
        
        off_dk = t * stride_dk_t + offs_d * stride_dk_d
        tl.store(dk_base + off_dk, dkr, mask=mask_d)
        tl.store(dk_base + off_dk + 1, dki, mask=mask_d)
        
        # -------------------------------------------------
        # E. Backtrack Accumulator State
        # -------------------------------------------------
        # We move to t-1. We must remove V(t)*K(t) from Acc.
        term_vk_r = vr * kr - vi * ki
        term_vk_i = vr * ki + vi * kr
        
        acc_r -= term_vk_r
        acc_i -= term_vk_i