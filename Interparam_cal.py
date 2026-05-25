import torch
import torch.nn as nn


class Interaction_parameter_cal(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.thermo_pre_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 8)
        )

        self.sub_pre_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, symmetric_out):
        # symmetric_out: [batch_size, seq_len (x + y), hidden_size]
        symmetric_out = symmetric_out.mean(dim=1)
        symmetric_out = self.thermo_pre_head(symmetric_out)
        return {
            'polarity_compatibility': symmetric_out[:, 0],
            'size_compatibility': symmetric_out[:, 1],
            'hbond_compatibility': symmetric_out[:, 2],
            'hydrophobicity_compatibility': symmetric_out[:, 3],
            'electrostatic_compatibility': symmetric_out[:, 4],
            'flexibility_compatibility': symmetric_out[:, 5],
            'charge_compatibility': symmetric_out[:, 7],
        }