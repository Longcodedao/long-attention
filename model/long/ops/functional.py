import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Tuple, List


def parallel_scan(k, v, gate, gamma):
    """
    Stable Parallel Scan for Linear Attention (Real-Valued).
    Computes: h_t = gamma * h_{t-1} + (k_t * v_t * gate_t)
    
    Uses Log-Space Cumulative Sum for numerical stability on GPU.
    """
    # 1. Compute Input: u_t = k * v * gate
    u = k * v * gate
    
    # 2. Prepare Decay Mask
    B, T, H, D = u.shape
    # Create time indices [0, 1, ..., T-1]
    t_steps = torch.arange(T, device=u.device).view(1, T, 1, 1).float()
    
    # 3. Compute Decay Curve: gamma^t
    # gamma shape [1, 1, H, 1] broadcasts to [1, T, H, 1]
    decay_curve = torch.pow(gamma, t_steps)
    
    # 4. Numerical Stability Trick:
    # h_t = gamma^t * Sum(u_i / gamma^i)
    
    # Inverse decay: 1 / gamma^t
    decay_inv = 1.0 / (decay_curve + 1e-9)
    
    # Cumulative sum of "undecayed" inputs
    u_undecayed = torch.cumsum(u * decay_inv, dim=1)
    
    # Re-apply decay
    h = u_undecayed * decay_curve
    
    return h


@torch.jit.script 
def recurrent_scan(k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor, 
                   gamma: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused Recurrent Scan for Inference (or short sequences).
    Computes: h_t = gamma_t * h_{t-1} + (k_t * v_t * gate_t)
    """
    B, T, H, D = k.shape
    
    # Pre-compute inputs [B, T, H, D]
    u = k * v * gate 
    
    # Ensure gamma is correctly shaped for broadcasting [B, T, H, 1]
    # In recurrent loop, we slice by T
    
    current_state = state
    output_list: List[torch.Tensor] = []

    for t in range(T):
        # Slice current time step
        u_t = u[:, t]          # [B, H, D]
        gamma_t = gamma[:, t]  # [B, H, 1] (Broadcasting over D)
        
        # Recurrence: h_t = h_{t-1} * gamma_t + u_t
        current_state = (current_state * gamma_t) + u_t
        output_list.append(current_state)

    # Stack: [B, T, H, D]
    outputs = torch.stack(output_list, dim=1)
    return outputs, current_state

    
def chunked_parallel_scan(k, v, gate, gamma, chunk_size=128):
    """
    Stable Chunked Parallel Scan.
    """
    B, T, H, D = k.shape

    # 1. Padding
    pad_len = (chunk_size - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        gate = F.pad(gate, (0, 0, 0, 0, 0, pad_len))
        # Pad gamma with 0.0 (full decay) or 1.0 (no decay)? 
        # Using 1.0 preserves state flow, but K/V are 0, so no new info added.
        gamma = F.pad(gamma, (0, 0, 0, 0, 0, pad_len), value=1.0)

    T_padded = T + pad_len
    num_chunks = T_padded // chunk_size

    # 2. View as Chunks
    k_chunk = k.view(B, num_chunks, chunk_size, H, D)
    v_chunk = v.view(B, num_chunks, chunk_size, H, D)
    gate_chunk = gate.view(B, num_chunks, chunk_size, H, D)
    gamma_chunk = gamma.view(B, num_chunks, chunk_size, H, 1)

    # 3. Local Scan (Log-Space for Stability)
    # Cast to float64 for precision
    u_chunk = (k_chunk * v_chunk * gate_chunk).to(torch.float32)
    gamma_chunk = gamma_chunk.to(torch.float32)

    # Cumsum of log(gamma)
    log_gamma = torch.log(gamma_chunk + 1e-10)
    S = torch.cumsum(log_gamma, dim=2) # [B, N, C, H, 1]
    
    # Local Scan Formula: 
    # h_t = exp(S_t) * cumsum( u_t * exp(-S_t) )
    u_prime = u_chunk * torch.exp(-S)
    scan_prime = torch.cumsum(u_prime, dim=2)
    h_intra = scan_prime * torch.exp(S)

    # 4. Inter-Chunk Recurrence (The Carry)
    # Total decay for a block is exp(S_last)
    chunk_decay = torch.exp(S[:, :, -1]) # [B, N, H, 1]
    chunk_end_states = h_intra[:, :, -1] # [B, N, H, D]
    
    carry_states = []
    last_carry = torch.zeros(B, H, D, device=k.device, dtype=torch.float32)

    for i in range(num_chunks):
        carry_states.append(last_carry)
        # Update carry: Carry_next = Carry_curr * Block_Decay + Block_End_State
        current_decay = chunk_decay[:, i] # [B, H, 1]
        current_end = chunk_end_states[:, i]
        last_carry = (last_carry * current_decay) + current_end

    # Stack carries [B, N, 1, H, D] for broadcasting
    carry_states = torch.stack(carry_states, dim=1).unsqueeze(2)

    # 5. Fuse Global + Local
    # h_final = h_intra + (Carry * decay_profile)
    # decay_profile is exp(S) relative to start of block
    global_term = carry_states * torch.exp(S)
    h_final = global_term + h_intra

    # 6. Cleanup
    h_final = h_final.view(B, T_padded, H, D)
    if pad_len > 0:
        h_final = h_final[:, :T]
    
    return h_final.to(k.dtype)