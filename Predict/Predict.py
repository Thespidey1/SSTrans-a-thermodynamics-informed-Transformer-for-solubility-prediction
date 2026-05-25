import os
import random

import numpy as np

import yaml
import pandas as pd
from torch.utils.data import DataLoader
from dataset import dataset_generation
from torch.amp import autocast
import time
from tokenizer import SMILES_Atomwise_Tokenizer
from models import Whole_block
import torch

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        tensor = tensor.float()
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn_labeled(input_batch):
    starter_time = time.time()
    # Extract all elements from the batch
    Solute_SMILES = [item[0] for item in input_batch]
    Solvent_SMILES = [item[1] for item in input_batch]
    solubility = [item[2] for item in input_batch]
    temperature = [item[3] for item in input_batch]
    T_ref = [item[4] for item in input_batch]
    Solute_ref = [item[5] for item in input_batch]
    Solvent_ref = [item[6] for item in input_batch]

    solvation_free_energy = [item[7] for item in input_batch]
    solvation_enthalpy = [item[8] for item in input_batch]
    heat_capacity_cp = [item[9] for item in input_batch]
    heat_capacity_cs = [item[10] for item in input_batch]
    sublimation_enthalpy = [item[11] for item in input_batch]

    polarity_compatibility = [item[12] for item in input_batch]
    size_compatibility = [item[13] for item in input_batch]
    hbond_compatibility = [item[14] for item in input_batch]
    hydrophobicity_compatibility = [item[15] for item in input_batch]
    electrostatic_compatibility = [item[16] for item in input_batch]
    flexibility_compatibility = [item[17] for item in input_batch]
    aromaticity_compatibility = [item[18] for item in input_batch]
    charge_compatibility = [item[19] for item in input_batch]

    solubility_gradient = [item[20] for item in input_batch]
    result = {}

    # Process SMILES data with None checks
    if None not in Solute_SMILES:
        tokenized_Solute_SMILES = tokenizer(
            Solute_SMILES,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        result["Solute_SMILES_ids"] = tokenized_Solute_SMILES["input_ids"]
        result["Solute_attention_mask"] = tokenized_Solute_SMILES["attention_mask"]
    else:
        result["Solute_SMILES_ids"] = None
        result["Solute_attention_mask"] = None

    if None not in Solvent_SMILES:
        tokenized_Solvent_SMILES = tokenizer(
            Solvent_SMILES,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        result["Solvent_SMILES_ids"] = tokenized_Solvent_SMILES["input_ids"]
        result["Solvent_attention_mask"] = tokenized_Solvent_SMILES["attention_mask"]
    else:
        result["Solvent_SMILES_ids"] = None
        result["Solvent_attention_mask"] = None

    if None not in Solute_ref:
        tokenized_Solute_ref = tokenizer(
            Solute_ref,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        result["Solute_ref"] = tokenized_Solute_ref["input_ids"]
        result["Solute_ref_mask"] = tokenized_Solute_ref["attention_mask"]
    else:
        result["Solute_ref"] = None
        result["Solute_ref_mask"] = None

    if None not in Solvent_ref:
        tokenized_Solvent_ref = tokenizer(
            Solvent_ref,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True
        )
        result["Solvent_ref"] = tokenized_Solvent_ref["input_ids"]
        result["Solvent_ref_mask"] = tokenized_Solvent_ref["attention_mask"]
    else:
        result["Solvent_ref"] = None
        result["Solvent_ref_mask"] = None

    def process_tensor(values, key_name):
        if None not in values:
            tensor = torch.tensor(values)
            tensor = tensor.unsqueeze(1)
            # Normalize if we have a normalizer for this feature
            if key_name in normalizers:
                tensor = normalizers[key_name].norm(tensor)
            result[key_name] = tensor
        else:
            result[key_name] = None

    if None not in temperature:
        temperature = torch.tensor(temperature)
        temperature = temperature.unsqueeze(1)
        temperature_norm = norm_temp(temperature)
        result["temperature"] = temperature
        result["temperature_norm"] = temperature_norm
    else:
        result["temperature"] = None
        result["temperature_norm"] = None
    if None not in T_ref:
        T_ref = torch.tensor(T_ref)
        T_ref = T_ref.unsqueeze(1)
        T_ref_norm = norm_temp(T_ref)
        result["T_ref"] = T_ref
        result["T_ref_norm"] = T_ref_norm
    else:
        result["T_ref"] = None
        result["T_ref_norm"] = None
    # Process all numerical properties
    process_tensor(solubility, "solubility")
    process_tensor(solvation_free_energy, "solvation_free_energy")
    process_tensor(solvation_enthalpy, "solvation_enthalpy")
    process_tensor(heat_capacity_cp, "heat_capacity_cp")
    process_tensor(heat_capacity_cs, "heat_capacity_cs")
    process_tensor(sublimation_enthalpy, "sublimation_enthalpy")
    process_tensor(polarity_compatibility, "polarity_compatibility")
    process_tensor(size_compatibility, "size_compatibility")
    process_tensor(hbond_compatibility, "hbond_compatibility")
    process_tensor(hydrophobicity_compatibility, "hydrophobicity_compatibility")
    process_tensor(electrostatic_compatibility, "electrostatic_compatibility")
    process_tensor(flexibility_compatibility, "flexibility_compatibility")
    process_tensor(aromaticity_compatibility, "aromaticity_compatibility")
    process_tensor(charge_compatibility, "charge_compatibility")
    process_tensor(solubility_gradient, "solubility_gradient")

    ender_time = time.time()
    print(f"Total collate time: {ender_time - starter_time:.4f} seconds")

    return result

def norm_temp(tensor):
    return 1 / tensor - 1 / 298


def denorm_temp(normed_tensor):
    return 1 / (normed_tensor + 1 / 298)

if __name__ == '__main__':
    config = yaml.load(open("config.yaml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    model_args = config["transformer"]
    tokenizer = SMILES_Atomwise_Tokenizer('vocab.txt')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    pre_set = pd.read_excel('Smiles_for_pre.xlsx', sheet_name=0)
    dataset_pre = dataset_generation(pre_set)
    pre_loader = DataLoader(
        dataset_pre,
        batch_size=1024,
        collate_fn=collate_fn_labeled,
        shuffle=False,
        pin_memory=True,
    )

    try:
        normalizer_states = torch.load('test_stage_norm.pt', map_location=device)
        normalizers = {}
        for feature, state in normalizer_states.items():
            normalizers[feature] = Normalizer(torch.tensor([0.0]))
            normalizers[feature].load_state_dict(state)
        print("Loaded existing normalizers")
    except Exception as e:
        raise RuntimeError(f"No normalizers or failed to load test_stage_norm.pt: {e}")

    model_paths = [f'best_model{i}.pth' for i in range(1, 6)]
    model = Whole_block(**model_args).to(device)

    all_model_solubility = []

    for model_idx, model_path in enumerate(model_paths, start=1):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        print(f"Loading {model_path} ...")
        model_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(model_dict)
        model.eval()

        model_solubility_batches = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(pre_loader):
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    predictions = model(
                        batch["Solute_SMILES_ids"].to(device),
                        batch["Solvent_SMILES_ids"].to(device),
                        batch["Solute_attention_mask"].to(device),
                        batch["Solvent_attention_mask"].to(device),
                        batch["temperature"].to(device)
                    )

                solubility = predictions['solubility']
                solubility = normalizers['solubility'].denorm(solubility)
                model_solubility_batches.append(solubility.detach().cpu())

        model_solubility = torch.cat(model_solubility_batches, dim=0).flatten()
        all_model_solubility.append(model_solubility)
        print(f"Finished {model_path}, prediction count: {len(model_solubility)}")

    # Shape: [num_samples, 5]
    all_model_solubility = torch.stack(all_model_solubility, dim=1)
    mean_solubility = all_model_solubility.mean(dim=1)

    df_solubility = pd.DataFrame({
        'sample_id': range(len(mean_solubility)),
        'model1_predicted_solubility': all_model_solubility[:, 0].numpy(),
        'model2_predicted_solubility': all_model_solubility[:, 1].numpy(),
        'model3_predicted_solubility': all_model_solubility[:, 2].numpy(),
        'model4_predicted_solubility': all_model_solubility[:, 3].numpy(),
        'model5_predicted_solubility': all_model_solubility[:, 4].numpy(),
        'mean_predicted_solubility': mean_solubility.numpy(),
    })

    output_path = 'solubility_predictions.xlsx'
    df_solubility.to_excel(output_path, index=False)
