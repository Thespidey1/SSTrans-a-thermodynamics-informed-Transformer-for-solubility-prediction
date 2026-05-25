import torch
import torch.nn as nn
from transformers import RobertaConfig, RobertaModel, RobertaForMaskedLM
from Multihead_attention import Multi_Symmetric_CrossAttention, Multi_Asymmetric_CrossAttention
from Thermodynamic_modules import Thermodynamic_parameter_cal, Thermodynamic_Mapping
from Interparam_cal import Interaction_parameter_cal
import torch.nn.functional as F


class Roberta_encoder(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float, unsupervised_pretrained_path=None):
        super().__init__()

        config = RobertaConfig(
            vocab_size=ntoken,
            hidden_size=d_model,
            num_hidden_layers=nlayers,
            num_attention_heads=nhead,
            intermediate_size=d_hid,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            output_attentions=False,
            max_position_embeddings=1024,
        )
        self.roberta = RobertaModel(config)

        if unsupervised_pretrained_path:
            print(f"Loading pretrained weights from {unsupervised_pretrained_path}")
            pretrained_state = torch.load(unsupervised_pretrained_path, map_location='cpu')

            if 'state_dict' in pretrained_state:
                pretrained_dict = pretrained_state['state_dict']
            elif 'model' in pretrained_state:
                pretrained_dict = pretrained_state['model']
            elif 'roberta' in pretrained_state:
                pretrained_dict = pretrained_state['roberta']
            else:
                pretrained_dict = pretrained_state

            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

            model_dict = self.roberta.state_dict()

            matched_dict = {}
            for name, param in pretrained_dict.items():
                # 移除 'roberta.' 前缀
                if name.startswith('roberta.'):
                    new_name = name[8:]
                else:
                    new_name = name

                if new_name.startswith('lm_head'):
                    print(f"Skipping {name} (lm_head layer not needed)")
                    continue
                if new_name == 'embeddings.position_embeddings.weight':
                    print(f"Skipping {name} (position_embeddings layer not needed)")
                    continue

                if new_name not in model_dict:
                    print(f"Skipping {name} -> {new_name} (not in current model)")
                    continue

                if model_dict[new_name].shape != param.shape:
                    print(f"Skipping {name} -> {new_name} due to shape mismatch: "
                          f"pretrained {param.shape} vs current {model_dict[new_name].shape}")
                    continue

                matched_dict[new_name] = param
                print(f"Loaded {name} -> {new_name}")

            print(f"Successfully loaded {len(matched_dict)}/{len(model_dict)} layers from pretrained model")

            if len(matched_dict) > 0:
                model_dict.update(matched_dict)
                load_result = self.roberta.load_state_dict(model_dict, strict=True)

                if load_result.missing_keys:
                    print("Missing keys:", load_result.missing_keys)
                if load_result.unexpected_keys:
                    print("Unexpected keys:", load_result.unexpected_keys)
            else:
                print("Warning: No weights were loaded from the pretrained model!")

    def forward(self, x):
        attention_mask = (x != 0).long()
        output = self.roberta(input_ids=x.long(), attention_mask=attention_mask)
        return output


class Roberta_mlm(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float):
        super().__init__()

        config = RobertaConfig(
            vocab_size=ntoken,
            hidden_size=d_model,
            num_hidden_layers=nlayers,
            num_attention_heads=nhead,
            intermediate_size=d_hid,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            output_attentions=False,
            max_position_embeddings=1024,
        )
        self.roberta_mlm = RobertaForMaskedLM(config)

    def forward(self, x, labels):
        attention_mask = (x != 0).long()

        output = self.roberta_mlm(input_ids=x.long(), attention_mask=attention_mask, labels=labels)

        return output


class CNN(nn.Module):
    def __init__(self,
                 dropout,
                 embed_size,
                 num_filters=(100, 200, 200, 200, 200, 100, 100),
                 ngram_filter_sizes=(1, 2, 3, 4, 5, 6, 7)):
        super().__init__()

        self.num_filters = num_filters

        self.textcnn = nn.ModuleList([nn.Conv1d(in_channels=embed_size, out_channels=nf, kernel_size=ks)
                                      for nf, ks in zip(num_filters, ngram_filter_sizes)])
        self.dropout_cnn = nn.Dropout(dropout)
        self.output = nn.Linear(sum(num_filters), embed_size)

    def forward(self, src_nd):
        encoded = src_nd.permute(0, 2, 1)

        textcnn_out = [F.relu(conv(encoded)) for conv in self.textcnn]
        textcnn_out = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in textcnn_out]  # Max pooling
        textcnn_out = torch.cat(textcnn_out, 1)  # Concatenate all the pooled features
        # input_vecs = torch.cat((IL_textcnn_out, T.view(-1, 1), P.view(-1, 1)), dim=1)
        input_vecs = textcnn_out
        # input_vecs = torch.cat((self.lin(IL_textcnn_out), T.view(-1, 1)), dim=1)
        input_vecs = self.dropout_cnn(input_vecs)
        out = self.output(input_vecs.float())

        # out = self.output(IL_encoded[:,0,:].float())

        return out


class Simple_pre_head(nn.Module):
    def __init__(self, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.solubility_pre_head = nn.Sequential(
            nn.Linear(hidden_size + 1, hidden_size // 2),
            nn.Dropout(dropout),
            nn.Softplus(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, concat_out, temp):
        concat_out = concat_out.mean(dim=1)  # 先池化再MLP
        concat_out = torch.concat([concat_out, temp], dim=1)
        out = self.solubility_pre_head(concat_out)

        return out


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout):
        super().__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.gelu = nn.GELU()
        self.dropout_f = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        x = self.dropout_f(x)
        return x


class Symmetry_cross_attention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num, intermediate_size, dropout):
        super().__init__()

        self.cross_att = Multi_Symmetric_CrossAttention(hidden_size, all_head_size, head_num, dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)

    def forward(self, x_o, y_o, attention_mask_1, attention_mask_2):
        output = self.cross_att(x_o, y_o, attention_mask_1, attention_mask_2)
        output = output + self.feed_forward(self.layer_norm(output))
        return output


class Asymmetry_cross_attention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num, intermediate_size, dropout):
        super().__init__()

        self.cross_att = Multi_Asymmetric_CrossAttention(hidden_size, all_head_size, head_num, dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)

    def forward(self, x_o, y_o, attention_mask_1, attention_mask_2):
        output = self.cross_att(x_o, y_o, attention_mask_1, attention_mask_2)
        output = output + self.feed_forward(self.layer_norm(output))
        return output


class Mere_symmetric_part(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float):
        super().__init__()

        self.Solute_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.Solvent_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.sym_cross_attention_block = Symmetry_cross_attention(d_model, d_model, nhead, d_hid, dropout)
        self.Interaction_parameter_cal = Interaction_parameter_cal(d_model, dropout)
        self.CNN = CNN(dropout, d_model)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.Dropout(dropout),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, y, attention_mask_1, attention_mask_2, temp):
        x = self.Solute_encoder(x)
        x = x.last_hidden_state
        y = self.Solvent_encoder(y)
        y = y.last_hidden_state
        sys_out = self.sym_cross_attention_block(x, y, attention_mask_1, attention_mask_2)
        inter_params = self.Interaction_parameter_cal(sys_out)
        cat = torch.concat([x, sys_out, y], dim=1)
        output = self.CNN(cat)
        output = self.pred_head(torch.concat([output, temp], dim=1))
        return {
            'solubility': output,
            'polarity_compatibility': inter_params['polarity_compatibility'].unsqueeze(-1),
            'size_compatibility': inter_params['size_compatibility'].unsqueeze(-1),
            'hbond_compatibility': inter_params['hbond_compatibility'].unsqueeze(-1),
            'hydrophobicity_compatibility': inter_params['hydrophobicity_compatibility'].unsqueeze(-1),
            'electrostatic_compatibility': inter_params['electrostatic_compatibility'].unsqueeze(-1),
            'flexibility_compatibility': inter_params['flexibility_compatibility'].unsqueeze(-1),
            'aromaticity_compatibility': inter_params['aromaticity_compatibility'].unsqueeze(-1),
            'charge_compatibility': inter_params['charge_compatibility'].unsqueeze(-1),
        }


class Whole_block(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float,
                 unsupervised_pretrained_path=None):
        super().__init__()

        self.Solute_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout,
                                              unsupervised_pretrained_path=unsupervised_pretrained_path)
        self.Solvent_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout,
                                               unsupervised_pretrained_path=unsupervised_pretrained_path)
        self.sym_cross_attention_block = Symmetry_cross_attention(d_model, d_model, nhead, d_hid, dropout)
        self.asym_cross_attention_block = Asymmetry_cross_attention(d_model, d_model, nhead, d_hid, dropout)
        self.Thermodynamic_parameter_cal = Thermodynamic_parameter_cal(d_model, dropout)
        self.Thermodynamic_Mapping = Thermodynamic_Mapping(d_model)
        self.Interaction_parameter_cal = Interaction_parameter_cal(d_model, dropout)
        self.CNN = CNN(dropout, d_model)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.Dropout(dropout),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, y, attention_mask_1, attention_mask_2, temp):
        x = self.Solute_encoder(x)
        x = x.last_hidden_state
        y = self.Solvent_encoder(y)
        y = y.last_hidden_state
        sys_out = self.sym_cross_attention_block(x, y, attention_mask_1, attention_mask_2)
        asys_out = self.asym_cross_attention_block(x, y, attention_mask_1, attention_mask_2)
        thermo_params = self.Thermodynamic_parameter_cal(asys_out, x)
        thermo_modulation = self.Thermodynamic_Mapping(thermo_params)
        inter_params = self.Interaction_parameter_cal(sys_out)
        cat = torch.concat([x, sys_out * thermo_modulation, y], dim=1)
        output = self.CNN(cat)
        output = self.pred_head(torch.concat([output, temp], dim=1))
        return {
            'solubility': output,
            'solvation_free_energy': thermo_params['solvation_free_energy'].unsqueeze(-1),
            'solvation_enthalpy': thermo_params['solvation_enthalpy'].unsqueeze(-1),
            'sublimation_enthalpy': thermo_params['sublimation_enthalpy'].unsqueeze(-1),
            'polarity_compatibility': inter_params['polarity_compatibility'].unsqueeze(-1),
            'size_compatibility': inter_params['size_compatibility'].unsqueeze(-1),
            'hbond_compatibility': inter_params['hbond_compatibility'].unsqueeze(-1),
            'hydrophobicity_compatibility': inter_params['hydrophobicity_compatibility'].unsqueeze(-1),
            'electrostatic_compatibility': inter_params['electrostatic_compatibility'].unsqueeze(-1),
            'flexibility_compatibility': inter_params['flexibility_compatibility'].unsqueeze(-1),
            'charge_compatibility': inter_params['charge_compatibility'].unsqueeze(-1),

        }


class AS_pretrain(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float,
                 unsupervised_pretrained_path=None):
        super().__init__()

        self.Solute_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout,
                                              unsupervised_pretrained_path=unsupervised_pretrained_path)
        self.Solvent_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout,
                                               unsupervised_pretrained_path=unsupervised_pretrained_path)
        self.asym_cross_attention_block = Asymmetry_cross_attention(d_model, d_model, nhead, d_hid, dropout)
        self.Thermodynamic_parameter_cal = Thermodynamic_parameter_cal(d_model, dropout)
        self.Interaction_parameter_cal = Interaction_parameter_cal(d_model, dropout)
        self.CNN = CNN(dropout, d_model)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.Dropout(dropout),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, y, attention_mask_1, attention_mask_2):
        x = self.Solute_encoder(x)
        x = x.last_hidden_state
        y = self.Solvent_encoder(y)
        y = y.last_hidden_state
        asys_out = self.asym_cross_attention_block(x, y, attention_mask_1, attention_mask_2)
        thermo_params = self.Thermodynamic_parameter_cal(asys_out, x)
        return {
            'solvation_free_energy': thermo_params['solvation_free_energy'].unsqueeze(-1),
            'solvation_enthalpy': thermo_params['solvation_enthalpy'].unsqueeze(-1),
        }

class S_pretrain(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float,
                 unsupervised_pretrained_path=None):
        super().__init__()

        self.Solute_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout,
                                              unsupervised_pretrained_path=unsupervised_pretrained_path)
        self.Solvent_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout,
                                               unsupervised_pretrained_path=unsupervised_pretrained_path)
        self.sym_cross_attention_block = Symmetry_cross_attention(d_model, d_model, nhead, d_hid, dropout)
        self.Interaction_parameter_cal = Interaction_parameter_cal(d_model, dropout)

    def forward(self, x, y, attention_mask_1, attention_mask_2, temp):
        x = self.Solute_encoder(x)
        x = x.last_hidden_state
        y = self.Solvent_encoder(y)
        y = y.last_hidden_state
        sys_out = self.sym_cross_attention_block(x, y, attention_mask_1, attention_mask_2)
        inter_params = self.Interaction_parameter_cal(sys_out)

        return {
            'polarity_compatibility': inter_params['polarity_compatibility'].unsqueeze(-1),
            'size_compatibility': inter_params['size_compatibility'].unsqueeze(-1),
            'hbond_compatibility': inter_params['hbond_compatibility'].unsqueeze(-1),
            'hydrophobicity_compatibility': inter_params['hydrophobicity_compatibility'].unsqueeze(-1),
            'electrostatic_compatibility': inter_params['electrostatic_compatibility'].unsqueeze(-1),
            'flexibility_compatibility': inter_params['flexibility_compatibility'].unsqueeze(-1),
            'aromaticity_compatibility': inter_params['aromaticity_compatibility'].unsqueeze(-1),
            'charge_compatibility': inter_params['charge_compatibility'].unsqueeze(-1),
        }


class Whole_block_no_CNN(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float):
        super().__init__()

        self.Solute_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.Solvent_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.sym_cross_attention_block = Symmetry_cross_attention(d_model, d_model, nhead, d_hid, dropout)
        self.asym_cross_attention_block = Asymmetry_cross_attention(d_model, d_model, nhead, d_hid, dropout)
        self.Thermodynamic_parameter_cal = Thermodynamic_parameter_cal(d_model, dropout)
        self.Thermodynamic_Mapping = Thermodynamic_Mapping(d_model)
        self.Interaction_parameter_cal = Interaction_parameter_cal(d_model, dropout)
        self.solubility_pre_head = Simple_pre_head(d_model, dropout)

    def forward(self, x, y, attention_mask_1, attention_mask_2, temp):
        x = self.Solute_encoder(x)
        x = x.last_hidden_state
        y = self.Solvent_encoder(y)
        y = y.last_hidden_state
        sys_out = self.sym_cross_attention_block(x, y, attention_mask_1, attention_mask_2)
        asys_out = self.asym_cross_attention_block(x, y, attention_mask_1, attention_mask_2)
        thermo_params = self.Thermodynamic_parameter_cal(asys_out, x)
        thermo_modulation = self.Thermodynamic_Mapping(thermo_params)
        inter_params = self.Interaction_parameter_cal(sys_out)
        cat = torch.concat([x, sys_out * thermo_modulation, y], dim=1)
        output = self.solubility_pre_head(cat, temp)
        return {
            'solubility': output,
            'solvation_free_energy': thermo_params['solvation_free_energy'].unsqueeze(-1),
            'solvation_enthalpy': thermo_params['solvation_enthalpy'].unsqueeze(-1),
            'sublimation_enthalpy': thermo_params['sublimation_enthalpy'].unsqueeze(-1),
            'polarity_compatibility': inter_params['polarity_compatibility'].unsqueeze(-1),
            'size_compatibility': inter_params['size_compatibility'].unsqueeze(-1),
            'hbond_compatibility': inter_params['hbond_compatibility'].unsqueeze(-1),
            'hydrophobicity_compatibility': inter_params['hydrophobicity_compatibility'].unsqueeze(-1),
            'electrostatic_compatibility': inter_params['electrostatic_compatibility'].unsqueeze(-1),
            'flexibility_compatibility': inter_params['flexibility_compatibility'].unsqueeze(-1),
            'aromaticity_compatibility': inter_params['aromaticity_compatibility'].unsqueeze(-1),
            'charge_compatibility': inter_params['charge_compatibility'].unsqueeze(-1),

        }


class No_cross(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float):
        super().__init__()

        self.Solute_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.Solvent_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.CNN = CNN(dropout, d_model)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.Dropout(dropout),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, y, attention_mask_1, attention_mask_2, temp):
        x = self.Solute_encoder(x)
        x = x.last_hidden_state
        y = self.Solvent_encoder(y)
        y = y.last_hidden_state
        cat = torch.concat([x, y], dim=1)
        output = self.CNN(cat)
        output = self.pred_head(torch.concat([output, temp], dim=1))
        return {
            'solubility': output
        }


class Only_Cross(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float):
        super().__init__()

        self.Solute_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.Solvent_encoder = Roberta_encoder(ntoken, d_model, nhead, d_hid, nlayers, dropout)
        self.sym_cross_attention_block = Symmetry_cross_attention(d_model, d_model, nhead, d_hid, dropout)
        self.asym_cross_attention_block = Asymmetry_cross_attention(d_model, d_model, nhead, d_hid, dropout)
        self.Thermodynamic_parameter_cal = Thermodynamic_parameter_cal(d_model, dropout)
        self.Thermodynamic_Mapping = Thermodynamic_Mapping(d_model)
        self.Interaction_parameter_cal = Interaction_parameter_cal(d_model, dropout)
        self.CNN = CNN(dropout, d_model)
        self.pred_head = nn.Sequential(
            nn.Linear(d_model + 1, d_model // 2),
            nn.Dropout(dropout),
            nn.Softplus(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x, y, attention_mask_1, attention_mask_2, temp):
        x = self.Solute_encoder(x)
        x = x.last_hidden_state
        y = self.Solvent_encoder(y)
        y = y.last_hidden_state
        sys_out = self.sym_cross_attention_block(x, y, attention_mask_1, attention_mask_2)
        asys_out = self.asym_cross_attention_block(x, y, attention_mask_1, attention_mask_2)
        thermo_params = self.Thermodynamic_parameter_cal(asys_out, x)
        thermo_modulation = self.Thermodynamic_Mapping(thermo_params)
        inter_params = self.Interaction_parameter_cal(sys_out)
        cat = sys_out * thermo_modulation
        output = self.CNN(cat)
        output = self.pred_head(torch.concat([output, temp], dim=1))
        return {
            'solubility': output,
            'solvation_free_energy': thermo_params['solvation_free_energy'].unsqueeze(-1),
            'solvation_enthalpy': thermo_params['solvation_enthalpy'].unsqueeze(-1),
            'sublimation_enthalpy': thermo_params['sublimation_enthalpy'].unsqueeze(-1),
            'polarity_compatibility': inter_params['polarity_compatibility'].unsqueeze(-1),
            'size_compatibility': inter_params['size_compatibility'].unsqueeze(-1),
            'hbond_compatibility': inter_params['hbond_compatibility'].unsqueeze(-1),
            'hydrophobicity_compatibility': inter_params['hydrophobicity_compatibility'].unsqueeze(-1),
            'electrostatic_compatibility': inter_params['electrostatic_compatibility'].unsqueeze(-1),
            'flexibility_compatibility': inter_params['flexibility_compatibility'].unsqueeze(-1),
            'aromaticity_compatibility': inter_params['aromaticity_compatibility'].unsqueeze(-1),
            'charge_compatibility': inter_params['charge_compatibility'].unsqueeze(-1),

        }
