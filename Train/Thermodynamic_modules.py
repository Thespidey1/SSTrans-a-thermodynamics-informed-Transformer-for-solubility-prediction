import torch
import torch.nn as nn


class Thermodynamic_parameter_cal(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.thermo_pre_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 2)
        )

        self.sub_pre_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 3)
        )

    def forward(self, asymmetric_out, solute_rep):
        # asymmetric_out: [batch_size, seq_len (x + y), hidden_size]
        # solute_rep: [batch_size, seq_len (x), hidden_size]
        asymmetric_out = asymmetric_out.mean(dim=1)  # 先池化再MLP
        asymmetric_out = self.thermo_pre_head(asymmetric_out)
        solute_rep = solute_rep.mean(dim=1)
        solute_output = self.sub_pre_head(solute_rep)
        return {
            'solvation_free_energy': asymmetric_out[:, 0],
            'solvation_enthalpy': asymmetric_out[:, 1],
            'heat_capacity_cp': solute_output[:, 0],
            'heat_capacity_cs': solute_output[:, 1],
            'sublimation_enthalpy': solute_output[:, 2]
        }


class Thermodynamic_Mapping(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.thermo_projection = nn.Sequential(
            nn.Linear(5, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )

        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

    def forward(self, thermo_params):

        combined_params = torch.cat([
            thermo_params['solvation_free_energy'].unsqueeze(1),
            thermo_params['solvation_enthalpy'].unsqueeze(1),
            thermo_params['heat_capacity_cp'].unsqueeze(1),
            thermo_params['heat_capacity_cs'].unsqueeze(1),
            thermo_params['sublimation_enthalpy'].unsqueeze(1)
        ], dim=1)

        thermo_hidden = self.thermo_projection(combined_params)

        attention_modulation = self.attention_gate(thermo_hidden)

        # [batch_size, 1, hidden_size] for multi-head attention
        return attention_modulation.unsqueeze(1)
