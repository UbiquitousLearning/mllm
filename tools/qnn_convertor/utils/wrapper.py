import torch
from torch import nn
from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionMlp
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from rotate import AutoOperation

class VisionMLPWrapper(nn.Module):
    def __init__(self, mlp: VisionMlp, 
                 hadamard_fc1: torch.Tensor = None, 
                 hadamard_fc2: torch.Tensor = None):
        super(VisionMLPWrapper, self).__init__()
        self.mlp = mlp
        
        # assuming all weights are on the same device
        self.device = mlp.fc1.weight.device
        self.dtype = mlp.fc2.weight.dtype
        
        self.rotate_fc1 = True if hadamard_fc1 is not None else False
        if self.rotate_fc1:
            AutoOperation.rotate_output(mlp.fc1, hadamard_fc1)
            self.register_buffer("hadamard_fc1_T", hadamard_fc1.T.to(self.device, dtype=self.dtype))
        
        self.rotate_fc2 = True if hadamard_fc2 is not None else False
        if self.rotate_fc2:
            AutoOperation.rotate_input(mlp.fc2, hadamard_fc2.T)
            self.register_buffer("hadamard_fc2", hadamard_fc2.to(self.device, dtype=self.dtype))
        

    def forward(self, x):
        up = self.mlp.fc1(x)
        
        # rotate back
        if self.rotate_fc1:
            up = up @ self.hadamard_fc1_T
        
        act = self.mlp.act(up)
        
        if self.rotate_fc2:
            # rotate
            act = act @ self.hadamard_fc2
        
        return self.mlp.fc2(act)
    

class MLPWrapper(nn.Module):
    def __init__(self, mlp: Qwen2MLP, 
                 hadamard_up: torch.Tensor = None, 
                 hadamard_gate: torch.Tensor = None,
                 hadamard_down: torch.Tensor = None):
        super(MLPWrapper, self).__init__()
        self.mlp = mlp
        
        # assuming all weights are on the same device
        self.device = mlp.up_proj.weight.device
        self.dtype = mlp.up_proj.weight.dtype
        
        self.rotate_up = True if hadamard_up is not None else False
        if self.rotate_up:
            AutoOperation.rotate_output(mlp.up_proj, hadamard_up)
            self.register_buffer("hadamard_up_T", hadamard_up.T.to(self.device, dtype=self.dtype))
            
        self.rotate_gate = True if hadamard_gate is not None else False
        if self.rotate_gate:
            AutoOperation.rotate_output(mlp.gate_proj, hadamard_gate)
            self.register_buffer("hadamard_gate_T", hadamard_gate.T.to(self.device, dtype=self.dtype))
        
        self.rotate_down = True if hadamard_down is not None else False
        if self.rotate_down:
            AutoOperation.rotate_input(mlp.down_proj, hadamard_down.T)
            self.register_buffer("hadamard_down", hadamard_down.to(self.device, dtype=self.dtype))
        

    def forward(self, x):
        up = self.mlp.up_proj(x)
        gate = self.mlp.gate_proj(x)
        
        # rotate back
        if self.rotate_up:
            up = up @ self.hadamard_up_T
            
        if self.rotate_gate:
            gate = gate @ self.hadamard_gate_T
        
        gated_output = up * self.mlp.act_fn(gate)
        
        if self.rotate_down:
            # rotate
            gated_output = gated_output @ self.hadamard_down
        
        return self.mlp.down_proj(gated_output)
    