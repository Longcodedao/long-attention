import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional, Tuple, List


def chunked_parallel_scan(k, v, gate, gamma, chunk_size = 128):
    """
    Chunked Parallel Scan (Numerically Stable for Infinite Sequences)

    Strategy:
    1. Split sequence into blocks (e.g., 128 tokens)
    2. Compute local scan inside each block (safe range)
    3. Compute 'carry' states between blocks.
    4. Add carry to local scans
    """

    # 1. Prepare Inputs
    # k, v: [B, T, H, D]
    B, T, H, D = k.shape

    # Pad T to be divisible by chunk_size if necessary
    pad_len = (chunk_len - (T % chunk_size)) % chunk_size
    if pad_len > 0:
        k = F.pad(k, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))
        gate = F.pad(gate, (0, 0, 0, 0, 0, pad_len))

        
    # 2. Reshape into Chunks
    # New shape: [B, Num_Chunks, Chunk_Size, H, D]
    T_padded = T + pad_len
    num_chunks = T_padded // chunk_size

    k_chunk = k.view(B, num_chunks, chunk_size, H, D)
    v_chunk = v.view(B, num_chunks, chunk_size, H, D)
    gate_chunk = gate.view(B, num_chunks, chunk_size, H, D)


    # 3. Compute Intra-Chunk Scan (Local Scan)
    # This uses the FLoat64 trick, but only for length = 128, so it's very safe
    # Pre-compute local inputs
    u_chunk = (k_chunk * v_chunk * gate_chunk).to(torch.float64)
    gamma_d = gamma.double().view(1, 1, 1, H, 1) # Broadcastable

    
    # Local time indices [0, 1, ..., 127]
    t_local = torch.arange(chunk_size, device=k.device, dtype=torch.float64)
    t_local = t_local.view(1, 1, chunk_size, 1, 1)

    # Local decays
    # shape: [1, 1, Chunk_Size, H, 1]
    local_decay = torch.pow(gamma_d, t_local)
    local_decay_inv = 1.0 / (local_decay + 1e-9)

    # Local CumSum (The "Parallel Scan" inside the chunk)
    # Sum(u * gamma^-t)
    # Shape [B, Num_Chunks, 1, H, D]
    u_scanned = torch.cumsum(u_chunk * local_decay_inv, dim = 2)
    # Re-apply decay: h_local = gamma^t * Sum(...)
    h_intra = u_scanned * local_decay

    
    # 4. Compute Inter-Chunk Recurrence (The "Carry")
    # We need to pass the final state of Block i to Block i+1.
    
    # The total decay across one full chunk (gamma^128)
    # Shape: [1, 1, 1, H, 1]
    chunk_decay = torch.pow(gamma_d, chunk_size).view(1, 1, H, 1) # [1, 1, H, 1]

    # The state at the END of each chunk
    # shape: [B, Num_Chunks, H, D]
    chunk_states = h_intra[:, :, -1]

    # Run a sequential scan on the chunk ends (Recurrence)
    # Since Num_Chunks is small (T=4096 -> 32 chunks), a loop is fast enough.
    carry_states = []
    last_carry = torch.zeros(B, H, D, device=k.device, dtype=torch.float64)

    for i in range(num_chunks):
        carry_states.append(last_carry)

        # Local history is the last state in the local chunk
        # Prev_history * Gamma^Chunk_size + Local_history = True History
        last_carry = last_carry * chunk_decay + chunk_states[:, i]

    carry_states = torch.stack(carry_states, dim = 1).unsqueeze(2)

    # 5. Fuse Global + Local info
    # Global_State[t] = (Carry_from_prev_block * Gamma^t_local) + Local_State[t]
    # Broadcast carry to all timesteps in chunk
    # carry_states: [B, N, 1, H, D]
    # local_decay:  [1, 1, C, H, 1]
    global_term = carry_states * local_decay
    
    # Add to local scan
    h_final = global_term + h_intra
    
    # 6. Reshape back and cleanup
    h_final = h_final.view(B, T_padded, H, D)
    if pad_len > 0:
        h_final = h_final[:, :T]
        
    return h_final.to(k.dtype)


@torch.jit.script 
def recurrent_scan(k: torch.Tensor, v: torch.Tensor, gate: torch.Tensor, 
                  gamma: torch.Tensor, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Universal Recurrent Scan (Works for both Prompt & Generation).
    JIT-compiled to fuse the loop into a single fast kernel.
    
    Args:
        k, v, gate: [B, T, H, D] - Input sequence
        gamma: [1, 1, H, 1]      - Decay rates
        state: [B, H, D]         - Previous memory state
    """
    B, T, H, D = k.shape

    # Pre-compute inputs
    u = k * v * gate      # [B, T, H, D]
    gamma_vec = gamma.view(1, H, 1)     # [1, H, 1]
    
    # We need to collect outputs if T > 1 (Prompt Phase)
    output_list: List[torch.Tensor] = []

    # JIT compiles this Python loop into a fast fused kernel
    current_state = state

    # Loop over time (T)
    for t in range(T):
        # h_t = gamma * h_{t - 1} + u_t
        current_state = current_state * gamma_vec + u[:, t]

        # We store the state to use for query projection later
        # (Or you can compute query output here immediately to save memory)
        output_list.append(current_state)

    # Stack outputs: [B, T, H, D]
    outputs = torch.stack(output_list, dim = 1)

    # Return:
    # 1. Sequence of states (for attention output)
    # 2. Final state (to save for the next generation step)
    return outputs, current_state
    