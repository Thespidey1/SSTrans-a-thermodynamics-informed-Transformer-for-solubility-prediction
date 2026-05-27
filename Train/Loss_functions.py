import torch
import torch.nn as nn
import torch.nn.functional as F


class Interaction_parameter_cal:
    def __init__(self, device, alpha, beta, gamma, delta, epsilon, normalizers=None):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.normalizers = normalizers

    def main_loss(self, outputs, ref_outputs, labels):
        main_loss = F.mse_loss(outputs['solubility'], labels['solubility'].to(self.device))
        return main_loss

    def temperature_gradient_loss(self, outputs, ref_outputs, labels):
        """Calculate loss based on the gradient of solubility with respect to temperature"""
        # Extract temperature and normalized temperature
        predicted_gradient = outputs['predicted_gradient'].to(self.device)

        # If gradient is provided in the labels, use it directly
        true_gradient = labels['solubility_gradient'].to(self.device)

        # Calculate MSE between predicted and true gradients
        gradient_loss = F.mse_loss(predicted_gradient, true_gradient)
        return gradient_loss

    def thermodynamic_consistency_loss(self, outputs, ref_outputs, labels):
        normalizers = self.normalizers
        norm_T = labels['temperature_norm'].to(self.device)
        predicted_solubility = outputs['solubility']
        solvation_free_energy = outputs['solvation_free_energy']
        solvation_enthalpy = outputs['solvation_enthalpy']
        sublimation_enthalpy = outputs['sublimation_enthalpy']

        ref_solubility = ref_outputs['solubility']
        ref_solvation_free_energy = ref_outputs['solvation_free_energy']

        if 'solubility' in normalizers:
            ref_solubility = normalizers['solubility'].denorm(ref_solubility)

        if 'solvation_free_energy' in normalizers:
            solvation_free_energy = normalizers['solvation_free_energy'].denorm(solvation_free_energy)
            ref_solvation_free_energy = normalizers['solvation_free_energy'].denorm(ref_solvation_free_energy)

        if 'solvation_enthalpy' in normalizers:
            solvation_enthalpy = normalizers['solvation_enthalpy'].denorm(solvation_enthalpy)

        if 'sublimation_enthalpy' in normalizers:
            sublimation_enthalpy = normalizers['sublimation_enthalpy'].denorm(sublimation_enthalpy)

        R = 8.314 / 4184  # kcal/(mol·K)

        # ref_solubility = ref_solubility.detach()

        theoretical_solubility_298 = ref_solubility - (solvation_free_energy - ref_solvation_free_energy) / (R * 298 * 2.3)
        theoretical_solubility = theoretical_solubility_298 - (solvation_enthalpy + sublimation_enthalpy) * norm_T / (R * 2.3)
        if 'solubility' in normalizers:
            theoretical_solubility = normalizers['solubility'].norm(theoretical_solubility)

        # theoretical_solubility = theoretical_solubility.detach()
        solubility_consistency = F.mse_loss(predicted_solubility, theoretical_solubility)

        return solubility_consistency

    def Thermo_params_loss(self, outputs, ref_outputs, labels):
        solvation_free_energy = outputs['solvation_free_energy']
        solvation_enthalpy = outputs['solvation_enthalpy']

        ex_solvation_free_energy = labels['solvation_free_energy'].to(self.device)
        ex_solvation_enthalpy = labels['solvation_enthalpy'].to(self.device)

        solvation_free_energy_loss = F.mse_loss(solvation_free_energy, ex_solvation_free_energy)
        solvation_enthalpy_loss = F.mse_loss(solvation_enthalpy, ex_solvation_enthalpy)

        Total_thermo_loss = solvation_free_energy_loss + solvation_enthalpy_loss

        return Total_thermo_loss

    def cosmo_params_loss(self, outputs, labels):
        total_loss = 0.0
        valid_count = 0

        if 'solvation_free_energy' in labels and 'solvation_free_energy' in outputs:
            ex_sfe = labels['solvation_free_energy'].to(self.device)
            pred_sfe = outputs['solvation_free_energy']

            valid_mask = ~torch.isnan(ex_sfe) & ~torch.isnan(pred_sfe)
            valid_mask = valid_mask.squeeze(-1)

            if valid_mask.any():
                valid_ex = ex_sfe[valid_mask]
                valid_pred = pred_sfe[valid_mask]
                batch_loss = F.mse_loss(valid_pred, valid_ex, reduction='sum')
                total_loss += batch_loss
                valid_count += valid_mask.sum().item()

        if 'solvation_enthalpy' in labels and 'solvation_enthalpy' in outputs:
            ex_sh = labels['solvation_enthalpy'].to(self.device)
            pred_sh = outputs['solvation_enthalpy']
            valid_mask = ~torch.isnan(ex_sh) & ~torch.isnan(pred_sh)
            valid_mask = valid_mask.squeeze(-1)

            if valid_mask.any():
                valid_ex = ex_sh[valid_mask]
                valid_pred = pred_sh[valid_mask]
                batch_loss = F.mse_loss(valid_pred, valid_ex, reduction='sum')
                total_loss += batch_loss
                valid_count += valid_mask.sum().item()

        if valid_count > 0:
            total_loss = total_loss / valid_count
        else:
            total_loss = torch.tensor(0.0, device=self.device)

        return total_loss

    def cosmo_params_loss_2(self, outputs, labels):
        solvation_free_energy = outputs['solvation_free_energy']

        ex_solvation_free_energy = labels['solvation_free_energy'].to(self.device)

        solvation_free_energy_loss = F.mse_loss(solvation_free_energy, ex_solvation_free_energy)

        return solvation_free_energy_loss

    def inter_params_loss(self, outputs, ref_outputs, labels):

        polarity_compatibility = outputs['polarity_compatibility']
        size_compatibility = outputs['size_compatibility']
        hbond_compatibility = outputs['hbond_compatibility']
        hydrophobicity_compatibility = outputs['hydrophobicity_compatibility']
        electrostatic_compatibility = outputs['electrostatic_compatibility']
        flexibility_compatibility = outputs['flexibility_compatibility']
        charge_compatibility = outputs['charge_compatibility']

        ex_polarity_compatibility = labels['polarity_compatibility'].to(self.device)
        ex_size_compatibility = labels['size_compatibility'].to(self.device)
        ex_hbond_compatibility = labels['hbond_compatibility'].to(self.device)
        ex_hydrophobicity_compatibility = labels['hydrophobicity_compatibility'].to(self.device)
        ex_electrostatic_compatibility = labels['electrostatic_compatibility'].to(self.device)
        ex_flexibility_compatibility = labels['flexibility_compatibility'].to(self.device)
        ex_charge_compatibility = labels['charge_compatibility'].to(self.device)

        polarity_compatibility_loss = F.mse_loss(polarity_compatibility, ex_polarity_compatibility)
        size_compatibility_loss = F.mse_loss(size_compatibility, ex_size_compatibility)
        hbond_compatibility_loss = F.mse_loss(hbond_compatibility, ex_hbond_compatibility)
        hydrophobicity_compatibility_loss = F.mse_loss(hydrophobicity_compatibility, ex_hydrophobicity_compatibility)
        electrostatic_compatibility_loss = F.mse_loss(electrostatic_compatibility, ex_electrostatic_compatibility)
        flexibility_compatibility_loss = F.mse_loss(flexibility_compatibility, ex_flexibility_compatibility)
        charge_compatibility_loss = F.mse_loss(charge_compatibility, ex_charge_compatibility)

        Total_inter_loss = polarity_compatibility_loss + size_compatibility_loss + hbond_compatibility_loss + hydrophobicity_compatibility_loss + electrostatic_compatibility_loss + flexibility_compatibility_loss + charge_compatibility_loss

        return Total_inter_loss

    def inter_params_loss_2(self, outputs, labels):

        polarity_compatibility = outputs['polarity_compatibility']
        size_compatibility = outputs['size_compatibility']
        hbond_compatibility = outputs['hbond_compatibility']
        hydrophobicity_compatibility = outputs['hydrophobicity_compatibility']
        electrostatic_compatibility = outputs['electrostatic_compatibility']
        flexibility_compatibility = outputs['flexibility_compatibility']

        ex_polarity_compatibility = labels['polarity_compatibility'].to(self.device)
        ex_size_compatibility = labels['size_compatibility'].to(self.device)
        ex_hbond_compatibility = labels['hbond_compatibility'].to(self.device)
        ex_hydrophobicity_compatibility = labels['hydrophobicity_compatibility'].to(self.device)
        ex_electrostatic_compatibility = labels['electrostatic_compatibility'].to(self.device)
        ex_flexibility_compatibility = labels['flexibility_compatibility'].to(self.device)

        polarity_compatibility_loss = F.mse_loss(polarity_compatibility, ex_polarity_compatibility)
        size_compatibility_loss = F.mse_loss(size_compatibility, ex_size_compatibility)
        hbond_compatibility_loss = F.mse_loss(hbond_compatibility, ex_hbond_compatibility)
        hydrophobicity_compatibility_loss = F.mse_loss(hydrophobicity_compatibility, ex_hydrophobicity_compatibility)
        electrostatic_compatibility_loss = F.mse_loss(electrostatic_compatibility, ex_electrostatic_compatibility)
        flexibility_compatibility_loss = F.mse_loss(flexibility_compatibility, ex_flexibility_compatibility)

        Total_inter_loss = polarity_compatibility_loss + size_compatibility_loss + hbond_compatibility_loss + hydrophobicity_compatibility_loss + electrostatic_compatibility_loss + flexibility_compatibility_loss

        return Total_inter_loss

    def whole_loss(self, outputs, ref_outputs, labels):
        main_loss = self.main_loss(outputs, ref_outputs, labels)
        thermodynamic_consistency_loss = self.thermodynamic_consistency_loss(outputs, ref_outputs, labels)
        Thermo_params_loss = self.Thermo_params_loss(outputs, ref_outputs, labels)
        inter_params_loss = self.inter_params_loss(outputs, ref_outputs, labels)
        # temp_gradient_loss = self.temperature_gradient_loss(outputs, ref_outputs, labels)

        whole_loss = (main_loss + self.beta * thermodynamic_consistency_loss +
                      self.gamma * Thermo_params_loss + self.delta * inter_params_loss)

        return whole_loss

    def basic_loss(self, outputs, ref_outputs, labels):
        main_loss = self.main_loss(outputs, ref_outputs, labels)
        thermodynamic_consistency_loss = self.thermodynamic_consistency_loss(outputs, ref_outputs, labels)

        basic_loss = (main_loss + self.beta * thermodynamic_consistency_loss)

        return basic_loss

    def main_cons_therm_loss(self, outputs, ref_outputs, labels):
        main_loss = self.main_loss(outputs, ref_outputs, labels)
        thermodynamic_consistency_loss = self.thermodynamic_consistency_loss(outputs, ref_outputs, labels)
        Thermo_params_loss = self.Thermo_params_loss(outputs, ref_outputs, labels)

        main_cons_therm_loss = (main_loss + self.beta * thermodynamic_consistency_loss +
                                self.gamma * Thermo_params_loss)

        return main_cons_therm_loss
