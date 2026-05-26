from __future__ import print_function, division
import functools
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from joblib import Parallel, delayed


class dataset_generation(Dataset):
    def __init__(self, df):
        # 检查列是否存在，不存在则提供空列表作为默认值
        self.SoluteSMILES = df['SoluteSMILES'].tolist() if 'SoluteSMILES' in df.columns else []
        self.SolventSMILES = df['SolventSMILES'].tolist() if 'SolventSMILES' in df.columns else []
        self.label = df['meanSolubility'].values.tolist() if 'meanSolubility' in df.columns else []
        self.T_round = df['T_round'].values.tolist() if 'T_round' in df.columns else []
        self.T_ref = df['T_ref'].values.tolist() if 'T_ref' in df.columns else []
        self.Solute_ref = df['SoluteSMILES'].tolist() if 'SoluteSMILES' in df.columns else []
        self.Solvent_ref = df['Solvent_ref'].tolist() if 'Solvent_ref' in df.columns else []

        self.solvation_free_energy = df[
            'solvation_free_energy'].values.tolist() if 'solvation_free_energy' in df.columns else []
        self.solvation_enthalpy = df['solvation_enthalpy'].values.tolist() if 'solvation_enthalpy' in df.columns else []
        self.heat_capacity_cp = df['heat_capacity_cp'].values.tolist() if 'heat_capacity_cp' in df.columns else []
        self.heat_capacity_cs = df['heat_capacity_cs'].values.tolist() if 'heat_capacity_cs' in df.columns else []
        self.sublimation_enthalpy = df[
            'sublimation_enthalpy'].values.tolist() if 'sublimation_enthalpy' in df.columns else []

        self.polarity_compatibility = df[
            'polarity_compatibility'].values.tolist() if 'polarity_compatibility' in df.columns else []
        self.size_compatibility = df['size_compatibility'].values.tolist() if 'size_compatibility' in df.columns else []
        self.hbond_compatibility = df[
            'hbond_compatibility'].values.tolist() if 'hbond_compatibility' in df.columns else []
        self.hydrophobicity_compatibility = df[
            'hydrophobicity_compatibility'].values.tolist() if 'hydrophobicity_compatibility' in df.columns else []
        self.electrostatic_compatibility = df[
            'electrostatic_compatibility'].values.tolist() if 'electrostatic_compatibility' in df.columns else []
        self.flexibility_compatibility = df[
            'flexibility_compatibility'].values.tolist() if 'flexibility_compatibility' in df.columns else []
        self.aromaticity_compatibility = df[
            'aromaticity_compatibility'].values.tolist() if 'aromaticity_compatibility' in df.columns else []
        self.charge_compatibility = df[
            'charge_compatibility'].values.tolist() if 'charge_compatibility' in df.columns else []
        self.solubility_gradient = df[
            'solubility_gradient'].values.tolist() if 'solubility_gradient' in df.columns else []

        # 确保所有列表至少有一个元素，以防止空数据集
        self.data_length = max(
            len(self.SoluteSMILES),
            len(self.SolventSMILES),
            len(self.label),
            len(self.T_round),
            len(self.T_ref),
            len(self.Solute_ref),
            len(self.Solvent_ref),
            len(self.solvation_free_energy),
            len(self.solvation_enthalpy),
            len(self.heat_capacity_cp),
            len(self.heat_capacity_cs),
            len(self.sublimation_enthalpy),
            len(self.polarity_compatibility),
            len(self.size_compatibility),
            len(self.hbond_compatibility),
            len(self.hydrophobicity_compatibility),
            len(self.electrostatic_compatibility),
            len(self.flexibility_compatibility),
            len(self.aromaticity_compatibility),
            len(self.charge_compatibility),
            len(self.solubility_gradient),
        )

        # 如果数据集完全为空（所有列都不存在），则设置长度为0
        if self.data_length == 0:
            self.data_length = 0

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # 处理可能的索引越界情况
        if index >= self.data_length:
            raise IndexError(f"Index {index} out of range for dataset with length {self.data_length}")

        # 安全获取各项数据，如果列表为空或索引超出范围则返回None
        SoluteSMILES = self.SoluteSMILES[index] if index < len(self.SoluteSMILES) else None
        SolventSMILES = self.SolventSMILES[index] if index < len(self.SolventSMILES) else None
        meanSolubility = self.label[index] if index < len(self.label) else None
        temp = self.T_round[index] if index < len(self.T_round) else None

        T_ref = self.T_ref[index] if index < len(self.T_ref) else None
        Solute_ref = self.Solute_ref[index] if index < len(self.Solute_ref) else None
        Solvent_ref = self.Solvent_ref[index] if index < len(self.Solvent_ref) else None
        solvation_free_energy = self.solvation_free_energy[index] if index < len(self.solvation_free_energy) else None
        solvation_enthalpy = self.solvation_enthalpy[index] if index < len(self.solvation_enthalpy) else None
        heat_capacity_cp = self.heat_capacity_cp[index] if index < len(self.heat_capacity_cp) else None
        heat_capacity_cs = self.heat_capacity_cs[index] if index < len(self.heat_capacity_cs) else None
        sublimation_enthalpy = self.sublimation_enthalpy[index] if index < len(self.sublimation_enthalpy) else None

        polarity_compatibility = self.polarity_compatibility[index] if index < len(
            self.polarity_compatibility) else None
        size_compatibility = self.size_compatibility[index] if index < len(self.size_compatibility) else None
        hbond_compatibility = self.hbond_compatibility[index] if index < len(self.hbond_compatibility) else None
        hydrophobicity_compatibility = self.hydrophobicity_compatibility[index] if index < len(
            self.hydrophobicity_compatibility) else None
        electrostatic_compatibility = self.electrostatic_compatibility[index] if index < len(
            self.electrostatic_compatibility) else None
        flexibility_compatibility = self.flexibility_compatibility[index] if index < len(
            self.flexibility_compatibility) else None
        aromaticity_compatibility = self.aromaticity_compatibility[index] if index < len(
            self.aromaticity_compatibility) else None
        charge_compatibility = self.charge_compatibility[index] if index < len(self.charge_compatibility) else None
        solubility_gradient = self.solubility_gradient[index] if index < len(self.solubility_gradient) else None

        # 统一返回结构
        return SoluteSMILES, SolventSMILES, meanSolubility, temp, T_ref, Solute_ref, Solvent_ref, \
            solvation_free_energy, solvation_enthalpy, heat_capacity_cp, heat_capacity_cs, sublimation_enthalpy, \
            polarity_compatibility, size_compatibility, hbond_compatibility, hydrophobicity_compatibility, \
            electrostatic_compatibility, flexibility_compatibility, aromaticity_compatibility, charge_compatibility, solubility_gradient


class dataset_generation_mlm(Dataset):
    def __init__(self, df):
        # 检查列是否存在，不存在则提供空列表作为默认值
        self.SMILES = df['canonical_smiles'].tolist() if 'canonical_smiles' in df.columns else []

        # 设置数据集长度
        self.data_length = len(self.SMILES)

    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        # 处理可能的索引越界情况
        if index >= self.data_length:
            raise IndexError(f"Index {index} out of range for dataset with length {self.data_length}")

        # 安全获取SMILES数据
        SMILES = self.SMILES[index] if index < len(self.SMILES) else None

        # 统一返回结构
        return SMILES
