import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LongAttention(nn.Module):
    """
    LongAttention: O(N) Linear Attention via Holographic Binding.
    
    Paper: "Holo-Transformer: A Hybrid Architecture for O(N) Long-Context Reasoning"
    
    Args:
        dim (int): Input dimension (d_model).
        hd_dim (int): Hidden dimension for holographic binding (default: 8192).
        max_seq_len (int): Maximum sequence length for positional buffer.
    """
    def __init__(self, dim, hd_dim=8192, max_seq_len=131072):
        super().__init__()
        self.dim = dim
        self.hd_dim = hd_dim
        
        # 1. Projections
        # Input -> [Content_Phase, Content_Mag, Query_Real, Query_Imag]
        # We use a single linear layer for efficiency
        self.proj_in = nn.Linear(dim, hd_dim * 4, bias=False)
        self.proj_out = nn.Linear(hd_dim * 2, dim, bias=False)
        
        # 2. Positional Memory Binding (Random Fourier Features)
        # We use Random Fourier frequencies to ensure orthogonality across dimensions.
        # This fixes the "Neighbor Leakage" problem of standard RoPE.
        # omega ~ Uniform(0, 2pi)
        # phase[t, k] = t * omega[k]
        
        # Use a fixed generator for determinism (Traceability)
        rng = torch.Generator()
        rng.manual_seed(42)
        
        # Generate random frequencies (Uniform [0, 2pi] guarantees high orthogonality)
        omega = torch.rand(1, hd_dim, generator=rng) * 2 * math.pi
        
        # Pre-compute phases: t * omega
        t = torch.arange(max_seq_len).unsqueeze(1).float()
        phases = t * omega # [MaxLen, HD]
        
        # Register as a buffer so it saves with the model but isn't a learned parameter
        self.register_buffer('pos_phases', phases, persistent=True)
        
        # 3. Normalization
        self.out_norm = nn.LayerNorm(hd_dim * 2)
        
        # Init weights
        nn.init.normal_(self.proj_in.weight, std=0.02)
        nn.init.normal_(self.proj_out.weight, std=0.02)

    def forward(self, x):
        """
        Args:
            x: Input tensor [Batch, SeqLen, Dim]
        Returns:
            Output tensor [Batch, SeqLen, Dim]
        """
        B, T, C = x.shape
        
        # 1. Project Input (BF16/FP16 safe)
        # [B, T, HD*4]
        raw = self.proj_in(x)
        c_phase, c_mag, q_real, q_imag = raw.chunk(4, dim=-1)
        
        # 2. Construct Components
        # Content Phasor: Amplitude * exp(i * Phase)
        # Sigmoid magnitude ensures bounded energy. Tanh phase maps to [-pi, pi].
        content = torch.sigmoid(c_mag) * torch.exp(1j * (torch.tanh(c_phase) * math.pi))
        
        # Positional Phasor: exp(i * PosPhase)
        # Slicing the pre-computed buffer: pos_phases is [MaxLen, HD] -> [1, T, HD]
        pos_p = torch.exp(1j * self.pos_phases[:T]).unsqueeze(0) 
        
        # 3. Holographic Binding
        # Key = Content * Position (Rotation in Complex Plane)
        # This is kept in input dtype (BF16/FP16) temporarily to save memory bandwidth
        key = content * pos_p 
        
        # --- CRITICAL STABILITY BLOCK ---
        # We must cast to Float32 (Complex64) for the Prefix Scan (CumSum).
        # This prevents "swamping" (loss of precision) where early tokens vanish in long sequences.
        key_f32 = key.to(torch.complex64) 
        
        # 4. Linear Memory Accumulation (The O(N) Scan)
        # Memory[t] = Memory[t-1] + Key[t]
        # This effectively stores the superposed history at every step.
        memory_state = torch.cumsum(key_f32, dim=1)
        
        # 5. Dynamic Query Generation
        # The query vector determines "what" or "where" to retrieve.
        query = torch.complex(q_real, q_imag).to(torch.complex64) # Match precision
        
        # 6. Retrieval (Unbinding)
        # Value = Memory * Query_Conjugate (Inverse Rotation)
        # If Query approx equals Pos[k], this unbinds Content[k].
        retrieval = memory_state * query.conj()
        
        # 7. Readout & Cast Back
        # Convert Complex -> Real (Concatenate Real/Imag parts)
        ret_real = torch.view_as_real(retrieval).flatten(-2)
        
        # Cast back to original input dtype (BF16/FP16) for the rest of the network
        ret_real = ret_real.to(x.dtype)
        
        # Final Projection
        ret_real = self.out_norm(ret_real)
        return self.proj_out(ret_real)