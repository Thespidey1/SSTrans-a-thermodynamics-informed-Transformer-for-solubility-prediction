import pickle
import gc
import bisect
import lz4
import torch
import yaml
import threading
import platform
from collections import OrderedDict
import os
import csv
import sys
import time
import multiprocessing as mp
import pandas as pd
import torch.nn as nn
from transformers.data.data_collator import DataCollatorForLanguageModeling
import rdkit
import numpy as np
import h5py
from torch.utils.data import DataLoader, Dataset
from dataset import dataset_generation, dataset_generation_mlm
from rdkit import Chem
import contextlib
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem import MACCSkeys
from torch.utils.data import Subset
from sklearn.model_selection import KFold
from tqdm import tqdm
from torch.amp import GradScaler, autocast
import random
import time
import math
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tokenizer import SMILES_Atomwise_Tokenizer
from Mlm_data_collator import SafeDimensionDataCollator
from models import Roberta_encoder, Roberta_mlm, Whole_block, No_cross, Whole_block_no_CNN, Only_Cross, AS_pretrain, S_pretrain
from Loss_functions import Interaction_parameter_cal
import torch
from torch.utils.data import Dataset, IterableDataset, random_split
import os
import mmap
from typing import Optional, Dict, List
import json

def setup_windows_optimizations():
    torch.set_num_threads(os.cpu_count())
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
    print(f'Windows optimizations applied: {os.cpu_count()} threads')

def _get_chunk_size_standalone(chunk_path):
    chunk_file = os.path.basename(chunk_path)
    if chunk_file.endswith('.pt'):
        try:
            chunk_data = torch.load(chunk_path, map_location='cpu', mmap=True)
            chunk_size = len(chunk_data)
            del chunk_data
        except:
            chunk_data = torch.load(chunk_path)
            chunk_size = len(chunk_data)
            del chunk_data
    elif chunk_file.endswith('.lz4'):
        with open(chunk_path, 'rb') as f:
            compressed_data = f.read()
            decompressed = lz4.frame.decompress(compressed_data)
            chunk_data = pickle.loads(decompressed)
            chunk_size = len(chunk_data)
            del chunk_data, compressed_data, decompressed
    else:
        with open(chunk_path, 'rb') as f:
            chunk_data = pickle.load(f)
            chunk_size = len(chunk_data)
            del chunk_data
    gc.collect()
    return chunk_size

def get_optimal_workers():
    cpu_count = os.cpu_count()
    if platform.system() == 'Windows':
        optimal_workers = max(1, cpu_count - 1)
        optimal_workers = min(optimal_workers, 32)
    else:
        optimal_workers = min(cpu_count, 32)
    print(f'System: {platform.system()}, CPU count: {cpu_count}, Using {optimal_workers} workers')
    return optimal_workers

class ChunkedPrecomputedDataset:

    def __init__(self, chunk_dir: str, max_cache_size: int=3, prefetch_next: bool=True, use_compression: bool=False, preload_metadata: bool=True, max_workers: int=None):
        self.chunk_dir = chunk_dir
        self.max_cache_size = max_cache_size
        self.prefetch_next = prefetch_next
        self.use_compression = use_compression
        if max_workers is None:
            self.max_workers = get_optimal_workers()
        else:
            self.max_workers = max_workers
        self.chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.endswith('.pt') or f.endswith('.pkl') or f.endswith('.lz4')])
        if not self.chunk_files:
            raise ValueError(f'No chunk files found in {chunk_dir}')
        self.chunk_cache = OrderedDict()
        self.cache_lock = threading.RLock()
        self.metadata_file = os.path.join(chunk_dir, 'metadata.pkl')
        if preload_metadata and os.path.exists(self.metadata_file):
            print('Loading cached metadata...')
            with open(self.metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self.chunk_offsets = metadata['offsets']
                self.chunk_sizes = metadata['sizes']
        else:
            print('Computing chunk metadata...')
            self.chunk_offsets, self.chunk_sizes = self._compute_metadata_optimized()
            if preload_metadata:
                with open(self.metadata_file, 'wb') as f:
                    pickle.dump({'offsets': self.chunk_offsets, 'sizes': self.chunk_sizes}, f)
        self.total_size = self.chunk_offsets[-1]
        self.prefetch_executor = None
        if self.prefetch_next:
            import concurrent.futures
            self.prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        print(f'Found {len(self.chunk_files)} chunk files')
        print(f'First few chunks: {self.chunk_files[:5]}')
        try:
            test_chunk = self._load_chunk(0)
            print(f'First chunk loaded successfully with {len(test_chunk)} samples')
        except Exception as e:
            print(f'Error loading first chunk: {e}')

    def _compute_metadata_optimized(self):
        num_cores = self.max_workers
        if platform.system() == 'Windows':
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass
        print(f'Computing metadata using {num_cores} processes...')
        try:
            with mp.Pool(processes=num_cores) as pool:
                chunk_paths = [os.path.join(self.chunk_dir, f) for f in self.chunk_files]
                sizes = list(tqdm(pool.imap(_get_chunk_size_standalone, chunk_paths), total=len(chunk_paths), desc=f'Computing metadata with {num_cores} processes'))
        except Exception as e:
            print(f'Multiprocessing failed: {e}, falling back to single process')
            chunk_paths = [os.path.join(self.chunk_dir, f) for f in self.chunk_files]
            sizes = [_get_chunk_size_standalone(path) for path in tqdm(chunk_paths, desc='Computing metadata (single process)')]
        offsets = [0]
        for size in sizes:
            offsets.append(offsets[-1] + size)
        return (offsets, sizes)

    def __len__(self):
        return self.total_size

    def _find_chunk_idx(self, idx: int) -> int:
        chunk_idx = bisect.bisect_right(self.chunk_offsets, idx) - 1
        return max(0, chunk_idx)

    def _load_chunk(self, chunk_idx: int) -> List:
        with self.cache_lock:
            if chunk_idx in self.chunk_cache:
                chunk_data = self.chunk_cache.pop(chunk_idx)
                self.chunk_cache[chunk_idx] = chunk_data
                return chunk_data
            chunk_file = self.chunk_files[chunk_idx]
            chunk_path = os.path.join(self.chunk_dir, chunk_file)
            try:
                if chunk_file.endswith('.lz4'):
                    with open(chunk_path, 'rb') as f:
                        compressed_data = f.read()
                        decompressed = lz4.frame.decompress(compressed_data)
                        chunk_data = pickle.loads(decompressed)
                elif chunk_file.endswith('.pt'):
                    chunk_data = torch.load(chunk_path, map_location='cpu')
                else:
                    with open(chunk_path, 'rb') as f:
                        chunk_data = pickle.load(f)
            except Exception as e:
                print(f'Error loading chunk {chunk_idx}: {e}')
                raise
            self.chunk_cache[chunk_idx] = chunk_data
            while len(self.chunk_cache) > self.max_cache_size:
                oldest_key = next(iter(self.chunk_cache))
                del self.chunk_cache[oldest_key]
            if self.prefetch_next and self.prefetch_executor:
                next_chunk_idx = chunk_idx + 1
                if next_chunk_idx < len(self.chunk_files):
                    if next_chunk_idx not in self.chunk_cache:
                        self.prefetch_executor.submit(self._prefetch_chunk, next_chunk_idx)
            return chunk_data

    def _prefetch_chunk(self, chunk_idx: int):
        try:
            with self.cache_lock:
                if chunk_idx not in self.chunk_cache and len(self.chunk_cache) < self.max_cache_size:
                    self._load_chunk(chunk_idx)
        except:
            pass

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_size:
            raise IndexError(f'Index {idx} out of range [0, {self.total_size})')
        chunk_idx = self._find_chunk_idx(idx)
        chunk_data = self._load_chunk(chunk_idx)
        local_idx = idx - self.chunk_offsets[chunk_idx]
        return chunk_data[local_idx]

    def get_chunk_info(self) -> Dict:
        return {'total_samples': self.total_size, 'num_chunks': len(self.chunk_files), 'avg_chunk_size': sum(self.chunk_sizes) / len(self.chunk_sizes), 'min_chunk_size': min(self.chunk_sizes), 'max_chunk_size': max(self.chunk_sizes), 'cache_size': len(self.chunk_cache), 'max_cache_size': self.max_cache_size}

    def clear_cache(self):
        with self.cache_lock:
            self.chunk_cache.clear()
            gc.collect()

    def preload_chunks(self, chunk_indices: List[int]):
        for chunk_idx in chunk_indices:
            if chunk_idx < len(self.chunk_files):
                self._load_chunk(chunk_idx)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('cache_lock', None)
        state.pop('prefetch_executor', None)
        state.pop('chunk_cache', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.chunk_cache = OrderedDict()
        self.cache_lock = threading.RLock()
        self.prefetch_executor = None
        if self.prefetch_next:
            import concurrent.futures
            self.prefetch_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def __del__(self):
        if hasattr(self, 'prefetch_executor') and self.prefetch_executor:
            self.prefetch_executor.shutdown(wait=False)

class MemoryMappedChunkedDataset(Dataset):

    def __init__(self, chunk_dir: str, mmap_dir: Optional[str]=None):
        self.chunk_dir = chunk_dir
        self.mmap_dir = mmap_dir or os.path.join(chunk_dir, 'mmap')
        self._create_memory_maps()

    def _create_memory_maps(self):
        import mmap
        os.makedirs(self.mmap_dir, exist_ok=True)
        chunk_files = sorted([f for f in os.listdir(self.chunk_dir) if f.endswith('.pt')])
        self.mmaps = []
        self.offsets = [0]
        for chunk_file in tqdm(chunk_files, desc='Creating memory maps'):
            chunk_path = os.path.join(self.chunk_dir, chunk_file)
            mmap_path = os.path.join(self.mmap_dir, chunk_file.replace('.pt', '.mmap'))
            if not os.path.exists(mmap_path):
                chunk_data = torch.load(chunk_path)
                with open(mmap_path, 'wb') as f:
                    pickle.dump(chunk_data, f)
                del chunk_data
            f = open(mmap_path, 'rb')
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            self.mmaps.append((f, mm))

    def __getitem__(self, idx):
        pass

    def __del__(self):
        for f, mm in self.mmaps:
            mm.close()
            f.close()

class PrecomputedDataset(Dataset):

    def __init__(self, precomputed_file):
        self.data = torch.load(precomputed_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn_precomputed(input_batch):
    result = {}
    all_fields = input_batch[0].keys()
    sequence_fields = {'SoluteSMILES_ids', 'SolventSMILES_ids', 'Solute_ref_ids', 'Solvent_ref_ids', 'Solute_ref', 'Solvent_ref'}
    mask_mapping = {'SoluteSMILES_ids': 'SoluteSMILES_mask', 'SolventSMILES_ids': 'SolventSMILES_mask', 'Solute_ref_ids': 'Solute_ref_mask', 'Solvent_ref_ids': 'Solvent_ref_mask', 'Solute_ref': 'Solute_ref_mask', 'Solvent_ref': 'Solvent_ref_mask'}
    for field in all_fields:
        values = [item[field] for item in input_batch if item[field] is not None]
        if len(values) == 0:
            result[field] = None
        elif isinstance(values[0], torch.Tensor) and values[0].dim() > 0:
            if field in sequence_fields:
                mask_field = mask_mapping.get(field)
                if mask_field and mask_field in all_fields:
                    mask_values = [item[mask_field] for item in input_batch if item[mask_field] is not None]
                    padded_ids = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)
                    padded_mask = torch.nn.utils.rnn.pad_sequence(mask_values, batch_first=True, padding_value=0)
                    result[field] = padded_ids
                    result[mask_field] = padded_mask
                else:
                    result[field] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)
            else:
                try:
                    result[field] = torch.stack(values)
                except RuntimeError as e:
                    if 'stack expects each tensor to be equal size' in str(e):
                        result[field] = torch.nn.utils.rnn.pad_sequence(values, batch_first=True, padding_value=0)
                    else:
                        raise e
        elif isinstance(values[0], torch.Tensor):
            result[field] = torch.stack(values)
        else:
            result[field] = values
    field_mapping = {'SoluteSMILES_ids': 'Solute_SMILES_ids', 'SoluteSMILES_mask': 'Solute_attention_mask', 'SolventSMILES_ids': 'Solvent_SMILES_ids', 'SolventSMILES_mask': 'Solvent_attention_mask', 'Solute_ref_ids': 'Solute_ref', 'Solute_ref_mask': 'Solute_ref_mask', 'Solvent_ref_ids': 'Solvent_ref', 'Solvent_ref_mask': 'Solvent_ref_mask'}
    for old_name, new_name in field_mapping.items():
        if old_name in result:
            result[new_name] = result.pop(old_name)
    return result

class ChunkedDatasetGeneration(IterableDataset):

    def __init__(self, file_path: str, chunksize: int=10000, total_lines: Optional[int]=None, is_train: bool=True, val_ratio: float=0.01, seed: int=42, max_workers: int=None):
        self.file_path = file_path
        self.chunksize = chunksize
        self.is_train = is_train
        self.val_ratio = val_ratio
        self.seed = seed
        self.max_workers = max_workers or mp.cpu_count()
        self._check_columns()
        if total_lines is None:
            print('Counting lines in file... This may take a while.')
            self.total_lines = self._count_lines_parallel()
        else:
            self.total_lines = total_lines
        np.random.seed(seed)
        self.val_indices = set(np.random.choice(self.total_lines, int(self.total_lines * val_ratio), replace=False))

    def _count_lines_parallel(self):

        def count_chunk_lines(chunk_info):
            start_pos, end_pos = chunk_info
            count = 0
            with open(self.file_path, 'rb') as f:
                f.seek(start_pos)
                while f.tell() < end_pos:
                    line = f.readline()
                    if not line:
                        break
                    count += 1
            return count
        file_size = os.path.getsize(self.file_path)
        chunk_size = file_size // self.max_workers
        chunks = []
        for i in range(self.max_workers):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < self.max_workers - 1 else file_size
            chunks.append((start, end))
        with mp.Pool(self.max_workers) as pool:
            results = pool.map(count_chunk_lines, chunks)
        return sum(results) - 1

    def _check_columns(self):
        first_chunk = pd.read_csv(self.file_path, nrows=1)
        self.available_columns = set(first_chunk.columns)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            effective_chunksize = self.chunksize * 2
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            effective_chunksize = self.chunksize
            worker_id = 0
            num_workers = 1
        chunks = pd.read_csv(self.file_path, chunksize=effective_chunksize, iterator=True, engine='c', low_memory=False)
        line_count = 0
        processed_chunks = 0
        for chunk_id, chunk in enumerate(chunks):
            if chunk_id % num_workers != worker_id:
                line_count += len(chunk)
                continue
            processed_chunks += 1
            valid_rows = []
            for idx, row in chunk.iterrows():
                if self.is_train and line_count in self.val_indices:
                    line_count += 1
                    continue
                elif not self.is_train and line_count not in self.val_indices:
                    line_count += 1
                    continue
                valid_rows.append(self._process_row(row))
                line_count += 1
                if len(valid_rows) >= 100:
                    for processed_row in valid_rows:
                        yield processed_row
                    valid_rows = []
            for processed_row in valid_rows:
                yield processed_row

    def _process_row(self, row):

        def safe_get(col_name):
            if hasattr(self, 'available_columns') and col_name not in self.available_columns:
                return None
            if col_name not in row:
                return None
            value = row[col_name]
            if pd.isna(value) or value == '' or value == 'None':
                return None
            if col_name not in ['SoluteSMILES', 'SolventSMILES', 'Solute_ref', 'Solvent_ref']:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
            return value
        solute_smiles = safe_get('SoluteSMILES')
        return (solute_smiles, safe_get('SolventSMILES'), safe_get('meanSolubility'), safe_get('T_round'), safe_get('T_ref'), solute_smiles, safe_get('Solvent_ref'), safe_get('solvation_free_energy'), safe_get('solvation_enthalpy'), safe_get('heat_capacity_cp'), safe_get('heat_capacity_cs'), safe_get('sublimation_enthalpy'), safe_get('polarity_compatibility'), safe_get('size_compatibility'), safe_get('hbond_compatibility'), safe_get('hydrophobicity_compatibility'), safe_get('electrostatic_compatibility'), safe_get('flexibility_compatibility'), safe_get('aromaticity_compatibility'), safe_get('charge_compatibility'), safe_get('solubility_gradient'))

class IndexedLargeFileDataset(Dataset):

    def __init__(self, file_path: str, index_path: Optional[str]=None, is_train: bool=True, val_ratio: float=0.01, seed: int=42):
        self.file_path = file_path
        self.is_train = is_train
        self.val_ratio = val_ratio
        self.file = open(file_path, 'rb')
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.mmap.seek(0)
        header_line = self.mmap.readline().decode('utf-8').strip()
        self.columns = [col.strip() for col in header_line.split(',')]
        self.available_columns = set(self.columns)
        self.column_indices = {col: idx for idx, col in enumerate(self.columns)}
        if index_path and os.path.exists(index_path):
            print(f'Loading index from {index_path}')
            with open(index_path, 'r') as f:
                self.index = json.load(f)
        else:
            print('Creating index... This will take time but only needs to be done once.')
            self.index = self._create_index()
            if index_path:
                with open(index_path, 'w') as f:
                    json.dump(self.index, f)
        np.random.seed(seed)
        total_lines = len(self.index['offsets'])
        val_size = int(total_lines * val_ratio)
        val_indices = set(np.random.choice(total_lines, val_size, replace=False))
        if is_train:
            self.indices = [i for i in range(total_lines) if i not in val_indices]
        else:
            self.indices = [i for i in range(total_lines) if i in val_indices]

    def _create_index(self):
        index = {'offsets': []}
        with open(self.file_path, 'rb') as f:
            f.readline()
            offset = f.tell()
            while True:
                line = f.readline()
                if not line:
                    break
                index['offsets'].append(offset)
                offset = f.tell()
        return index

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        self.mmap.seek(self.index['offsets'][actual_idx])
        line = self.mmap.readline().decode('utf-8', errors='ignore').strip()
        values = self._parse_csv_line(line)
        row_dict = {}
        for i, value in enumerate(values):
            if i < len(self.columns):
                row_dict[self.columns[i]] = value
        return self._process_row(row_dict)

    def _parse_csv_line(self, line):
        values = []
        current_value = []
        in_quotes = False
        for char in line:
            if char == '"':
                in_quotes = not in_quotes
            elif char == ',' and (not in_quotes):
                values.append(''.join(current_value).strip())
                current_value = []
            else:
                current_value.append(char)
        if current_value:
            values.append(''.join(current_value).strip())
        return values

    def _process_row(self, row_dict):

        def safe_get(col_name):
            if col_name not in self.available_columns:
                return None
            value = row_dict.get(col_name, None)
            if value is None or value == '' or value == 'nan' or (value == 'None'):
                return None
            if col_name not in ['SoluteSMILES', 'SolventSMILES', 'Solute_ref', 'Solvent_ref']:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return None
            return value
        solute_smiles = safe_get('SoluteSMILES')
        return (solute_smiles, safe_get('SolventSMILES'), safe_get('meanSolubility'), safe_get('T_round'), safe_get('T_ref'), solute_smiles, safe_get('Solvent_ref'), safe_get('solvation_free_energy'), safe_get('solvation_enthalpy'), safe_get('heat_capacity_cp'), safe_get('heat_capacity_cs'), safe_get('sublimation_enthalpy'), safe_get('polarity_compatibility'), safe_get('size_compatibility'), safe_get('hbond_compatibility'), safe_get('hydrophobicity_compatibility'), safe_get('electrostatic_compatibility'), safe_get('flexibility_compatibility'), safe_get('aromaticity_compatibility'), safe_get('charge_compatibility'), safe_get('solubility_gradient'))

    def __del__(self):
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file'):
            self.file.close()

def create_file_info(file_path):
    info_path = file_path + '.info'
    print('Analyzing file structure...')
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline().strip()
        columns = [col.strip() for col in header.split(',')]
    print('Counting lines...')
    total_lines = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        f.readline()
        for _ in f:
            total_lines += 1
    file_info = {'total_lines': total_lines, 'columns': columns, 'file_size': os.path.getsize(file_path)}
    with open(info_path, 'w') as f:
        json.dump(file_info, f, indent=2)
    print(f'File info saved to {info_path}')
    print(f'Total lines (excluding header): {total_lines}')
    print(f'Columns: {columns}')
    return file_info

def data_reading_large_file(config):
    if config['task_type'] == 'symmetric_pretraining':
        file_path = 'J:/molecule_solvent_combinations_with_interactions.txt'
        info_path = file_path + '.info'
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                file_info = json.load(f)
                total_lines = file_info.get('total_lines')
                print(f'Loaded file info: {total_lines} lines')
        else:
            print('No file info found. Creating...')
            file_info = create_file_info(file_path)
            total_lines = file_info['total_lines']
        if config.get('use_indexed_dataset', True):
            index_path = file_path + '.index'
            inter_train_set = IndexedLargeFileDataset(file_path, index_path, is_train=True, val_ratio=0.01, seed=config['seed'])
            inter_val_set = IndexedLargeFileDataset(file_path, index_path, is_train=False, val_ratio=0.01, seed=config['seed'])
        else:
            inter_train_set = ChunkedDatasetGeneration(file_path, chunksize=10000, total_lines=total_lines, is_train=True, val_ratio=0.01, seed=config['seed'])
            inter_val_set = ChunkedDatasetGeneration(file_path, chunksize=10000, total_lines=total_lines, is_train=False, val_ratio=0.01, seed=config['seed'])
        train_valid_set = inter_train_set
        return {'inter_train_set': inter_train_set, 'inter_val_set': inter_val_set, 'train_valid_set': train_valid_set}

def data_reading(config):
    if config['task_type'] == 'fine_tuning':
        train_valid_set = pd.read_excel('Fianlcleaned_elements_total_training_with_interactions.xlsx', sheet_name=0)
        train_set = pd.read_excel('Fianlcleaned_elements_training_with_interactions.xlsx', sheet_name=0)
        solute_valid_set = pd.read_excel('Solute_Validation_Set.xlsx', sheet_name=0)
        solvent_valid_set = pd.read_excel('Solvent_Validation_Set.xlsx', sheet_name=0)
        temperature_valid_set = pd.read_excel('Temperature_Validation_Set.xlsx', sheet_name=0)
        solute_set = pd.read_excel('Fianlcleaned_elements_solute_test_with_interactions.xlsx', sheet_name=0)
        solvent_set = pd.read_excel('Fianlcleaned_elements_solvent_test_with_interactions.xlsx', sheet_name=0)
        temperature_set = pd.read_excel('Fianlcleaned_elements_temperature_test_with_interactions.xlsx', sheet_name=0)
        return {'train_valid_set': train_valid_set, 'train_set': train_set, 'solute_valid_set': solute_valid_set, 'solvent_valid_set': solvent_valid_set, 'temperature_valid_set': temperature_valid_set, 'solute_set': solute_set, 'solvent_set': solvent_set, 'temperature_set': temperature_set}
    if config['task_type'] == 'asymmetric_pretraining':
        COSMO_set = pd.read_csv('CombiSolv-QM.csv')
        val_ratio = 0.01
        COSMO_size = len(COSMO_set)
        COSMO_val_size = int(COSMO_size * val_ratio)
        COSMO_val_set = COSMO_set.sample(COSMO_val_size, random_state=config['seed'])
        COSMO_train_set = COSMO_set.drop(COSMO_val_set.index)
        train_valid_set = COSMO_train_set
        return {'COSMO_train_set': COSMO_train_set, 'COSMO_val_set': COSMO_val_set, 'train_valid_set': train_valid_set}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class Normalizer(object):

    def __init__(self, tensor):
        tensor = tensor.float()
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean, 'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

def norm_temp(tensor):
    return 1 / tensor - 1 / 298

def denorm_temp(normed_tensor):
    return 1 / (normed_tensor + 1 / 298)

def percentage_within_threshold(y_true, y_pred, threshold):
    errors = np.abs(y_true - y_pred)
    return float(np.mean(errors <= threshold))

def dictionary_tokenized_generation(batch_smiles):
    tokenized = tokenizer(batch_smiles, padding=True, return_tensors='pt', return_attention_mask=True)
    batch_dicts = [{'input_ids': tokenized['input_ids'][i], 'attention_mask': tokenized['attention_mask'][i]} for i in range(len(tokenized['input_ids']))]
    return data_collator(batch_dicts)

def collate_fn_mlm(input_batch):
    SMILES = [item for item in input_batch]
    masked_SMILES = dictionary_tokenized_generation(SMILES)
    return {'smiles_input_ids': masked_SMILES['input_ids'], 'labels': masked_SMILES['labels']}

def collate_fn_medium(input_batch):
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
    if None not in Solute_SMILES:
        tokenized_Solute_SMILES = tokenizer(Solute_SMILES, padding=True, return_tensors='pt', return_attention_mask=True)
        result['Solute_SMILES_ids'] = tokenized_Solute_SMILES['input_ids']
        result['Solute_attention_mask'] = tokenized_Solute_SMILES['attention_mask']
    else:
        result['Solute_SMILES_ids'] = None
        result['Solute_attention_mask'] = None
    if None not in Solvent_SMILES:
        tokenized_Solvent_SMILES = tokenizer(Solvent_SMILES, padding=True, return_tensors='pt', return_attention_mask=True)
        result['Solvent_SMILES_ids'] = tokenized_Solvent_SMILES['input_ids']
        result['Solvent_attention_mask'] = tokenized_Solvent_SMILES['attention_mask']
    else:
        result['Solvent_SMILES_ids'] = None
        result['Solvent_attention_mask'] = None
    if None not in Solute_ref:
        tokenized_Solute_ref = tokenizer(Solute_ref, padding=True, return_tensors='pt', return_attention_mask=True)
        result['Solute_ref'] = tokenized_Solute_ref['input_ids']
        result['Solute_ref_mask'] = tokenized_Solute_ref['attention_mask']
    else:
        result['Solute_ref'] = None
        result['Solute_ref_mask'] = None
    if None not in Solvent_ref:
        tokenized_Solvent_ref = tokenizer(Solvent_ref, padding=True, return_tensors='pt', return_attention_mask=True)
        result['Solvent_ref'] = tokenized_Solvent_ref['input_ids']
        result['Solvent_ref_mask'] = tokenized_Solvent_ref['attention_mask']
    else:
        result['Solvent_ref'] = None
        result['Solvent_ref_mask'] = None

    def process_tensor_medium(values, key_name):
        if None not in values:
            tensor = torch.tensor(values)
            tensor = tensor.unsqueeze(1)
            result[key_name] = tensor
        else:
            result[key_name] = None
    if None not in temperature:
        temperature = torch.tensor(temperature)
        temperature = temperature.unsqueeze(1)
        temperature_norm = norm_temp(temperature)
        result['temperature'] = temperature
        result['temperature_norm'] = temperature_norm
    else:
        result['temperature'] = None
        result['temperature_norm'] = None
    if None not in T_ref:
        T_ref = torch.tensor(T_ref)
        T_ref = T_ref.unsqueeze(1)
        T_ref_norm = norm_temp(T_ref)
        result['T_ref'] = T_ref
        result['T_ref_norm'] = T_ref_norm
    else:
        result['T_ref'] = None
        result['T_ref_norm'] = None
    process_tensor_medium(solubility, 'solubility')
    process_tensor_medium(solvation_free_energy, 'solvation_free_energy')
    process_tensor_medium(solvation_enthalpy, 'solvation_enthalpy')
    process_tensor_medium(heat_capacity_cp, 'heat_capacity_cp')
    process_tensor_medium(heat_capacity_cs, 'heat_capacity_cs')
    process_tensor_medium(sublimation_enthalpy, 'sublimation_enthalpy')
    process_tensor_medium(polarity_compatibility, 'polarity_compatibility')
    process_tensor_medium(size_compatibility, 'size_compatibility')
    process_tensor_medium(hbond_compatibility, 'hbond_compatibility')
    process_tensor_medium(hydrophobicity_compatibility, 'hydrophobicity_compatibility')
    process_tensor_medium(electrostatic_compatibility, 'electrostatic_compatibility')
    process_tensor_medium(flexibility_compatibility, 'flexibility_compatibility')
    process_tensor_medium(aromaticity_compatibility, 'aromaticity_compatibility')
    process_tensor_medium(charge_compatibility, 'charge_compatibility')
    process_tensor_medium(solubility_gradient, 'solubility_gradient')
    return result

def collate_fn_labeled(input_batch):
    starter_time = time.time()
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
    if None not in Solute_SMILES:
        tokenized_Solute_SMILES = tokenizer(Solute_SMILES, padding=True, return_tensors='pt', return_attention_mask=True)
        result['Solute_SMILES_ids'] = tokenized_Solute_SMILES['input_ids']
        result['Solute_attention_mask'] = tokenized_Solute_SMILES['attention_mask']
    else:
        result['Solute_SMILES_ids'] = None
        result['Solute_attention_mask'] = None
    if None not in Solvent_SMILES:
        tokenized_Solvent_SMILES = tokenizer(Solvent_SMILES, padding=True, return_tensors='pt', return_attention_mask=True)
        result['Solvent_SMILES_ids'] = tokenized_Solvent_SMILES['input_ids']
        result['Solvent_attention_mask'] = tokenized_Solvent_SMILES['attention_mask']
    else:
        result['Solvent_SMILES_ids'] = None
        result['Solvent_attention_mask'] = None
    if None not in Solute_ref:
        tokenized_Solute_ref = tokenizer(Solute_ref, padding=True, return_tensors='pt', return_attention_mask=True)
        result['Solute_ref'] = tokenized_Solute_ref['input_ids']
        result['Solute_ref_mask'] = tokenized_Solute_ref['attention_mask']
    else:
        result['Solute_ref'] = None
        result['Solute_ref_mask'] = None
    if None not in Solvent_ref:
        tokenized_Solvent_ref = tokenizer(Solvent_ref, padding=True, return_tensors='pt', return_attention_mask=True)
        result['Solvent_ref'] = tokenized_Solvent_ref['input_ids']
        result['Solvent_ref_mask'] = tokenized_Solvent_ref['attention_mask']
    else:
        result['Solvent_ref'] = None
        result['Solvent_ref_mask'] = None

    def process_tensor(values, key_name):
        if None not in values:
            tensor = torch.tensor(values)
            tensor = tensor.unsqueeze(1)
            if key_name in normalizers:
                tensor = normalizers[key_name].norm(tensor)
            result[key_name] = tensor
        else:
            result[key_name] = None
    if None not in temperature:
        temperature = torch.tensor(temperature)
        temperature = temperature.unsqueeze(1)
        temperature_norm = norm_temp(temperature)
        result['temperature'] = temperature
        result['temperature_norm'] = temperature_norm
    else:
        result['temperature'] = None
        result['temperature_norm'] = None
    if None not in T_ref:
        T_ref = torch.tensor(T_ref)
        T_ref = T_ref.unsqueeze(1)
        T_ref_norm = norm_temp(T_ref)
        result['T_ref'] = T_ref
        result['T_ref_norm'] = T_ref_norm
    else:
        result['T_ref'] = None
        result['T_ref_norm'] = None
    process_tensor(solubility, 'solubility')
    process_tensor(solvation_free_energy, 'solvation_free_energy')
    process_tensor(solvation_enthalpy, 'solvation_enthalpy')
    process_tensor(heat_capacity_cp, 'heat_capacity_cp')
    process_tensor(heat_capacity_cs, 'heat_capacity_cs')
    process_tensor(sublimation_enthalpy, 'sublimation_enthalpy')
    process_tensor(polarity_compatibility, 'polarity_compatibility')
    process_tensor(size_compatibility, 'size_compatibility')
    process_tensor(hbond_compatibility, 'hbond_compatibility')
    process_tensor(hydrophobicity_compatibility, 'hydrophobicity_compatibility')
    process_tensor(electrostatic_compatibility, 'electrostatic_compatibility')
    process_tensor(flexibility_compatibility, 'flexibility_compatibility')
    process_tensor(aromaticity_compatibility, 'aromaticity_compatibility')
    process_tensor(charge_compatibility, 'charge_compatibility')
    process_tensor(solubility_gradient, 'solubility_gradient')
    ender_time = time.time()
    print(f'Total collate time: {ender_time - starter_time:.4f} seconds')
    return result

def metrics_compute(data_loader):
    all_predictions = []
    all_labels = []
    for batch in data_loader:
        label = batch['solubility']
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            predictions = model(batch['Solute_SMILES_ids'].to(device), batch['Solvent_SMILES_ids'].to(device), batch['Solute_attention_mask'].to(device), batch['Solvent_attention_mask'].to(device), batch['temperature'].to(device))
        predictions = predictions['solubility']
        all_predictions.append(predictions.cpu())
        all_labels.append(label.cpu())
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if 'solubility' in normalizers:
        all_predictions = normalizers['solubility'].denorm(all_predictions)
        all_labels = normalizers['solubility'].denorm(all_labels)
    pred_np = all_predictions.numpy()
    label_np = all_labels.numpy()
    r2 = r2_score(label_np, pred_np)
    rmse = math.sqrt(mean_squared_error(label_np, pred_np))
    percentage_in_1 = percentage_within_threshold(label_np, pred_np, 1.0)
    percentage_in_point7 = percentage_within_threshold(label_np, pred_np, 0.7)
    return (r2, rmse, percentage_in_1, percentage_in_point7, pred_np, label_np)

def metrics_compute_train(all_predictions, all_labels):
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    if 'solubility' in normalizers:
        all_predictions = normalizers['solubility'].denorm(all_predictions)
        all_labels = normalizers['solubility'].denorm(all_labels)
    pred_np = all_predictions.numpy()
    label_np = all_labels.numpy()
    r2 = r2_score(label_np, pred_np)
    rmse = math.sqrt(mean_squared_error(label_np, pred_np))
    percentage_in_1 = percentage_within_threshold(label_np, pred_np, 1.0)
    percentage_in_point7 = percentage_within_threshold(label_np, pred_np, 0.7)
    return (r2, rmse, percentage_in_1, percentage_in_point7, pred_np, label_np)

def freeze_layer(model, config):
    if config['Freeze']:
        for name, param in model.named_parameters():
            param.requires_grad = False
        if config['task_type'] == 'asymmetric_pretraining':
            print('Unfreezing asymmetric modules for pretraining')
            for name, param in model.asym_cross_attention_block.named_parameters():
                param.requires_grad = True
            for name, param in model.Thermodynamic_parameter_cal.named_parameters():
                param.requires_grad = True
        if config['task_type'] == 'symmetric_pretraining':
            print('Unfreezing symmetric modules for pretraining')
            for name, param in model.sym_cross_attention_block.named_parameters():
                param.requires_grad = True
            for name, param in model.Interaction_parameter_cal.named_parameters():
                param.requires_grad = True

def load_model(config, device, unsupervised_pretrained_path=None):
    model_args = config['transformer']
    if unsupervised_pretrained_path:
        model_args['unsupervised_pretrained_path'] = unsupervised_pretrained_path
    if config['model_name'] == 'AS_pretrain':
        model = AS_pretrain(**model_args).to(device)
    if config['model_name'] == 'S_pretrain':
        model = S_pretrain(**model_args).to(device)
    if config['model_name'] == 'whole_block':
        model = Whole_block(**model_args).to(device)
    elif config['model_name'] == 'no_cross':
        model = No_cross(**model_args).to(device)
    elif config['model_name'] == 'whole_block_no_CNN':
        model = Whole_block_no_CNN(**model_args).to(device)
    elif config['model_name'] == 'Only_Cross':
        model = Only_Cross(**model_args).to(device)
    total_params = sum((p.numel() for p in model.parameters()))
    loaded_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Total parameters: {total_params}, Trainable parameters: {loaded_params}')
    return model

def define_loss_function(config, output, ref_output, labels, normalizers):
    loss_function = Interaction_parameter_cal(device, alpha=config['alpha'], beta=config['beta'], gamma=config['gamma'], delta=config['delta'], epsilon=config['epsilon'], normalizers=normalizers)
    if config['loss_function'] == 'whole_loss':
        loss_function = loss_function.whole_loss(output, ref_output, labels)
    if config['loss_function'] == 'main':
        loss_function = loss_function.main_loss(output, ref_output, labels)
    if config['loss_function'] == 'basic_loss':
        loss_function = loss_function.basic_loss(output, ref_output, labels)
    if config['loss_function'] == 'main_cons_therm_loss':
        loss_function = loss_function.main_cons_therm_loss(output, ref_output, labels)
    return loss_function

def define_loss_function_2(config, output, labels, normalizers):
    loss_function = Interaction_parameter_cal(device, alpha=config['alpha'], beta=config['beta'], gamma=config['gamma'], delta=config['delta'], epsilon=config['epsilon'], normalizers=normalizers)
    if config['loss_function'] == 'cosmo_loss':
        loss_function = loss_function.cosmo_params_loss(output, labels)
    if config['loss_function'] == 'cosmo_loss_2':
        loss_function = loss_function.cosmo_params_loss_2(output, labels)
    if config['loss_function'] == 'inter_params_loss_2':
        loss_function = loss_function.inter_params_loss_2(output, labels)
    return loss_function

def create_normalizers(loader, features):
    print('Creating normalizers from training data...')
    normalizers = {}
    feature_values = {feature: [] for feature in features}
    for batch in tqdm(loader, desc='Collecting feature values'):
        for feature in features:
            if batch[feature] is not None:
                feature_values[feature].append(batch[feature])
    for feature in features:
        if feature_values[feature]:
            all_values_tensor = torch.cat(feature_values[feature], dim=0)
            normalizers[feature] = Normalizer(all_values_tensor)
    torch.save({f: n.state_dict() for f, n in normalizers.items()}, config('normalizer_path'))
    return normalizers
if __name__ == '__main__':
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU device: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.2f} GB')
    mp.set_start_method('spawn', force=True)
    features = ['solubility', 'solvation_free_energy', 'solvation_enthalpy', 'heat_capacity_cp', 'heat_capacity_cs', 'sublimation_enthalpy', 'polarity_compatibility', 'size_compatibility', 'hbond_compatibility', 'hydrophobicity_compatibility', 'electrostatic_compatibility', 'flexibility_compatibility', 'aromaticity_compatibility', 'charge_compatibility', 'solubility_gradient']
    config = yaml.load(open('config.yaml', 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    num_workers = max(1, mp.cpu_count() - 2)
    unsupervised_pretrain = config['unsupervised_pretrain']['pretrained_path']
    save_path = config['best_model_path']
    save_path_pretrain = config['save_path_pretrain']
    setup_seed(1)
    scaler = GradScaler()
    tokenizer = SMILES_Atomwise_Tokenizer('vocab.txt')
    data_collator = SafeDimensionDataCollator(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    batch_size = config['batch_size']
    if config['task_type'] == 'fine_tuning':
        if config['DA']:
            dataset_train_valid = PrecomputedDataset('Fianlcleaned_elements_total_training_with_interactions_precomputed_aug.pt')
        else:
            dataset_train_valid = PrecomputedDataset('Fianlcleaned_elements_total_training_with_interactions_precomputed.pt')
        dataset_train = PrecomputedDataset('Fianlcleaned_elements_training_with_interactions_precomputed.pt')
        dataset_solute_valid = PrecomputedDataset('Solute_Validation_Set_precomputed.pt')
        dataset_solvent_valid = PrecomputedDataset('Solvent_Validation_Set_precomputed.pt')
        dataset_temperature_valid = PrecomputedDataset('Temperature_Validation_Set_precomputed.pt')
        dataset_solute = PrecomputedDataset('Fianlcleaned_elements_solute_test_with_interactions_precomputed.pt')
        dataset_solvent = PrecomputedDataset('Fianlcleaned_elements_solvent_test_with_interactions_precomputed.pt')
        dataset_temperature = PrecomputedDataset('Fianlcleaned_elements_temperature_test_with_interactions_precomputed.pt')
        train_valid_loader_medium = DataLoader(dataset_train_valid, batch_size=batch_size, collate_fn=collate_fn_medium, shuffle=True, pin_memory=True)
        train_loader_medium = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate_fn_medium, shuffle=True, pin_memory=True)
        if config['model_stage'] == 'test':
            loader_mediun = train_valid_loader_medium
        elif config['model_stage'] == 'valid':
            loader_mediun = train_loader_medium
        try:
            normalizer_states = torch.load(config['normalizer_path'])
            normalizers = {}
            for feature, state in normalizer_states.items():
                normalizers[feature] = Normalizer(torch.tensor([0.0]))
                normalizers[feature].load_state_dict(state)
            print('Loaded existing normalizers')
        except:
            print('Will create new normalizers from training data')
            normalizers = create_normalizers(loader_mediun, features)
        train_valid_loader = DataLoader(dataset_train_valid, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=True, pin_memory=True)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=True, pin_memory=True)
        solute_valid_loader = DataLoader(dataset_solute_valid, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=True, pin_memory=True)
        solvent_valid_loader = DataLoader(dataset_solvent_valid, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=True, pin_memory=True)
        temperature_valid_loader = DataLoader(dataset_temperature_valid, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=True, pin_memory=True)
        if config['model_stage'] == 'test':
            loader = train_valid_loader
        if config['model_stage'] == 'valid':
            loader = train_loader
        solute_loader = DataLoader(dataset_solute, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=True, pin_memory=True)
        solvent_loader = DataLoader(dataset_solvent, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=True, pin_memory=True)
        temperature_loader = DataLoader(dataset_temperature, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=True, pin_memory=True)
        if config['model_stage'] == 'test':
            solute_loader = solute_loader
            solvent_loader = solvent_loader
            temperature_loader = temperature_loader
            train_valid_num = len(dataset_train_valid)
            solute_num = len(dataset_solute)
            solvent_num = len(dataset_solvent)
            temperature_num = len(dataset_temperature)
        if config['model_stage'] == 'valid':
            solute_loader = solute_valid_loader
            solvent_loader = solvent_valid_loader
            temperature_loader = temperature_valid_loader
            train_valid_num = len(dataset_train)
            solute_num = len(dataset_solute_valid)
            solvent_num = len(dataset_solvent_valid)
            temperature_num = len(dataset_solute_valid)
    if config['task_type'] == 'asymmetric_pretraining':
        if config['DA']:
            dataset_train_valid = PrecomputedDataset('CombiSolv-QM_precomputed_aug.pt')
        else:
            dataset_train_valid = PrecomputedDataset('CombiSolv-QM_precomputed.pt')
        train_ratio = 0.99
        test_ratio = 1 - train_ratio
        train_size = int(train_ratio * len(dataset_train_valid))
        test_size = len(dataset_train_valid) - train_size
        dataset_COSMO_train, dataset_COSMO_val = random_split(dataset_train_valid, [train_size, test_size], generator=torch.Generator().manual_seed(config['seed']))
        train_valid_loader_medium = DataLoader(dataset_train_valid, batch_size=batch_size, collate_fn=collate_fn_medium, shuffle=True, pin_memory=False)
        try:
            normalizer_states = torch.load(config['normalizer_path'])
            normalizers = {}
            for feature, state in normalizer_states.items():
                normalizers[feature] = Normalizer(torch.tensor([0.0]))
                normalizers[feature].load_state_dict(state)
            print('Loaded existing normalizers')
        except:
            print('Will create new normalizers from training data')
            normalizers = create_normalizers(loader_mediun, features)
        COSMO_train_loader = DataLoader(dataset_COSMO_train, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=True, pin_memory=False)
        COSMO_valid_loader = DataLoader(dataset_COSMO_val, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=True, pin_memory=False)
        loader_mediun = train_valid_loader_medium
        loader = COSMO_train_loader
        train_num = len(dataset_COSMO_train)
        valid_num = len(dataset_COSMO_val)
    if config['task_type'] == 'symmetric_pretraining':
        chunk_dir = 'D:/SSTrans/tokenized_data_chunks_optimized'
        optimal_num_workers = get_optimal_workers()
        if os.path.exists(chunk_dir) and len(os.listdir(chunk_dir)) > 0:
            print('Loading precomputed chunked data...')
            dataset_inter_train = ChunkedPrecomputedDataset(chunk_dir, max_cache_size=1, prefetch_next=False, use_compression=False, preload_metadata=True, max_workers=1)
            train_size = int(0.99 * len(dataset_inter_train))
            val_size = len(dataset_inter_train) - train_size
            dataset_inter_train, dataset_inter_val = random_split(dataset_inter_train, [train_size, val_size], generator=torch.Generator().manual_seed(config['seed']))
            print(f'Loaded {len(dataset_inter_train)} training samples and {len(dataset_inter_val)} validation samples')
        else:
            print('No precomputed data found. Using original file with dynamic tokenization...')
            datasets = data_reading_large_file(config)
            dataset_inter_train = datasets['inter_train_set']
            dataset_inter_val = datasets['inter_val_set']
        inter_train_loader = DataLoader(dataset_inter_train, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=False, pin_memory=False, num_workers=0, persistent_workers=False)
        inter_valid_loader = DataLoader(dataset_inter_val, batch_size=batch_size, collate_fn=collate_fn_precomputed, shuffle=False, pin_memory=False, num_workers=0, persistent_workers=False)
        loader = inter_train_loader
        train_num = len(dataset_inter_train)
        valid_num = len(dataset_inter_val)
        train_valid_loader_medium = DataLoader(dataset_inter_train, batch_size=batch_size, collate_fn=collate_fn_medium, shuffle=True, pin_memory=True)
        try:
            normalizer_states = torch.load(config['normalizer_path'])
            normalizers = {}
            for feature, state in normalizer_states.items():
                normalizers[feature] = Normalizer(torch.tensor([0.0]))
                normalizers[feature].load_state_dict(state)
            print('Loaded existing normalizers')
        except:
            print('Will create new normalizers from training data')
            normalizers = create_normalizers(loader_mediun, features)
        loader_mediun = train_valid_loader_medium
        print('=== DEBUGGING DATASET ACCESS ===')
        try:
            print('Test 1: Accessing first item directly...')
            start_time = time.time()
            first_item = dataset_inter_train[0]
            print(f'âœ“ Direct access successful in {time.time() - start_time:.2f}s')
            print(f'Item type: {type(first_item)}')
            if isinstance(first_item, dict):
                print(f'Keys: {list(first_item.keys())}')
        except Exception as e:
            print(f'âœ— Direct access failed: {e}')
            exit(1)
        try:
            print('\nTest 2: Testing raw iteration...')
            count = 0
            for item in dataset_inter_train:
                count += 1
                if count >= 3:
                    break
            print(f'âœ“ Raw iteration successful, tested {count} items')
        except Exception as e:
            print(f'âœ— Raw iteration failed: {e}')
            exit(1)
        try:
            print('\nTest 4: Testing with collate function...')
            test_loader = DataLoader(dataset_inter_train, batch_size=2, collate_fn=collate_fn_precomputed, shuffle=False, num_workers=0)
            start_time = time.time()
            test_batch = next(iter(test_loader))
            print(f'âœ“ Collate function works in {time.time() - start_time:.2f}s')
        except Exception as e:
            print(f'âœ— Collate function failed: {e}')
            print('The issue is with the collate_fn_precomputed function')
            exit(1)
        print('\n=== ALL TESTS PASSED - PROCEEDING WITH TRAINING ===')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using {} device.'.format(device))
    if config['metric_type'] == 'solubility':
        print('using {} molecules for training, {} molecules for solute extraplation, {} molecules for solvent extraplation,{} molecules for temperature extraplation.'.format(train_valid_num, solute_num, solvent_num, temperature_num))
        writer = {'train_rmse': SummaryWriter('./runs1/train_rmse'), 'solute_rmse': SummaryWriter('./runs1/solute_rmse'), 'solvent_rmse': SummaryWriter('./runs1/solvent_rmse'), 'temperature_rmse': SummaryWriter('./runs1/temperature_rmse'), 'train_r2': SummaryWriter('./runs1/train_r2'), 'solute_r2': SummaryWriter('./runs1/solute_r2'), 'solvent_r2': SummaryWriter('./runs1/solvent_r2'), 'temperature_r2': SummaryWriter('./runs1/temperature_r2'), 'train_log.7': SummaryWriter('./runs1/train_log.7'), 'solute_log.7': SummaryWriter('./runs1/solute_log.7'), 'solvent_log.7': SummaryWriter('./runs1/solvent_log.7'), 'temperature_log.7': SummaryWriter('./runs1/temperature_log.7'), 'train_log1': SummaryWriter('./runs1/train_log1'), 'solute_log1': SummaryWriter('./runs1/solute_log1'), 'solvent_log1': SummaryWriter('./runs1/solvent_log1'), 'temperature_log1': SummaryWriter('./runs1/temperature_log1')}
    if config['metric_type'] == 'cosmo_param':
        print('using {} molecules for training, {} molecules for validation'.format(train_num, valid_num))
        writer = {'loss_train': SummaryWriter('./runs1/loss_train'), 'loss_val': SummaryWriter('./runs1/loss_val')}
    if config['metric_type'] == 'inter_param':
        print('using {} molecules for training, {} molecules for validation'.format(train_num, valid_num))
        writer = {'loss_train': SummaryWriter('./runs1/loss_train'), 'loss_val': SummaryWriter('./runs1/loss_val')}
    model = load_model(config, device, unsupervised_pretrain)
    if config['asymmetric_pretrain']['pretrained_path']:
        asymmetric_pretrained_path = config['asymmetric_pretrain']['pretrained_path']
        asymmetric_pretrained_dict = torch.load(asymmetric_pretrained_path)
        model.asym_cross_attention_block.load_state_dict(asymmetric_pretrained_dict['asym_cross_attention_block'])
        model.Thermodynamic_parameter_cal.load_state_dict(asymmetric_pretrained_dict['Thermodynamic_parameter_cal'])
    if config['symmetric_pretrain']['pretrained_path']:
        symmetric_pretrained_path = config['symmetric_pretrain']['pretrained_path']
        symmetric_pretrained_dict = torch.load(symmetric_pretrained_path)
        model.sym_cross_attention_block.load_state_dict(symmetric_pretrained_dict['sym_cross_attention_block'])
        model.Interaction_parameter_cal.load_state_dict(symmetric_pretrained_dict['Interaction_parameter_cal'])
    freeze_layer(model, config)
    params = [p for p in model.parameters() if p.requires_grad]
    loaded_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'Trainable parameters (actaul): {loaded_params}')
    optimizer = optim.Adam(params, lr=config['init_lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=config['lr_decay_patience'], factor=config['lr_decay_factor'], min_lr=config['min_lr'])
    epochs = config['epochs']
    best_rmse = float('inf')
    best_loss = float('inf')
    model.train()
    running_loss = 0.0
    epoch_times = []
    early_stopping_count = 0
    if config['metric_type'] == 'solubility':
        for epoch in range(epochs):
            start_time = time.time()
            all_predictions = []
            all_labels = []
            for batch in loader:
                labels = {'solubility': batch['solubility'], 'temperature': batch['temperature'], 'temperature_norm': batch['temperature_norm'], 'solvation_free_energy': batch['solvation_free_energy'], 'solvation_enthalpy': batch['solvation_enthalpy'], 'heat_capacity_cp': batch['heat_capacity_cp'], 'heat_capacity_cs': batch['heat_capacity_cs'], 'sublimation_enthalpy': batch['sublimation_enthalpy'], 'polarity_compatibility': batch['polarity_compatibility'], 'size_compatibility': batch['size_compatibility'], 'hbond_compatibility': batch['hbond_compatibility'], 'hydrophobicity_compatibility': batch['hydrophobicity_compatibility'], 'electrostatic_compatibility': batch['electrostatic_compatibility'], 'flexibility_compatibility': batch['flexibility_compatibility'], 'aromaticity_compatibility': batch['aromaticity_compatibility'], 'charge_compatibility': batch['charge_compatibility']}
                solute_ids = batch['Solute_SMILES_ids'].to(device)
                solvent_ids = batch['Solvent_SMILES_ids'].to(device)
                solute_mask = batch['Solute_attention_mask'].to(device)
                solvent_mask = batch['Solvent_attention_mask'].to(device)
                temperature = batch['temperature'].to(device)
                solvent_ref = batch['Solvent_ref'].to(device) if batch['Solvent_ref'] is not None else None
                solvent_ref_mask = batch['Solvent_ref_mask'].to(device) if batch['Solvent_ref_mask'] is not None else None
                t_ref = batch['T_ref'].to(device) if batch['T_ref'] is not None else None
                optimizer.zero_grad()
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    output = model(solute_ids, solvent_ids, solute_mask, solvent_mask, temperature)
                    predictions = output['solubility']
                    label_solubility = batch['solubility']
                    all_predictions.append(predictions.cpu())
                    all_labels.append(label_solubility.cpu())
                    if solvent_ref is not None and solvent_ref_mask is not None and (t_ref is not None):
                        ref_output = model(solute_ids, solvent_ref, solute_mask, solvent_ref_mask, t_ref)
                    else:
                        ref_output = None
                    loss = define_loss_function(config, output, ref_output, labels, normalizers)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('max_grad_norm', 1.0))
                scaler.step(optimizer)
                scaler.update()
            with torch.no_grad():
                train_r2, train_rmse, train_1_percnetage, train_point7_percnetage, pred_train, label_train = metrics_compute_train(all_predictions, all_labels)
                model.eval()
                solute_r2, solute_rmse, solute_1_percnetage, solute_point7_percnetage, pred_solute, label_solute = metrics_compute(solute_loader)
                solvent_r2, solvent_rmse, solvent_1_percnetage, solvent_point7_percnetage, pred_solvent, label_solvent = metrics_compute(solvent_loader)
                temperature_r2, temperature_rmse, temperature_1_percnetage, temperature_point7_percnetage, pred_temperature, label_temperature = metrics_compute(temperature_loader)
                all_rmse = solute_rmse + solvent_rmse + temperature_rmse
                if config['csv_save']:
                    train_save_path = config['train_save_path']
                    df_train = pd.DataFrame({'pred': pred_train.flatten(), 'label': label_train.flatten()})
                    df_train.to_csv(train_save_path, index=False)
                    solute_save_path = config['solute_save_path']
                    df_solute = pd.DataFrame({'pred': pred_solute.flatten(), 'label': label_solute.flatten()})
                    df_solute.to_csv(solute_save_path, index=False)
                    solvent_save_path = config['solvent_save_path']
                    df_solvent = pd.DataFrame({'pred': pred_solvent.flatten(), 'label': label_solvent.flatten()})
                    df_solvent.to_csv(solvent_save_path, index=False)
                    temperature_save_path = config['temperature_save_path']
                    df_temperature = pd.DataFrame({'pred': pred_temperature.flatten(), 'label': label_temperature.flatten()})
                    df_temperature.to_csv(temperature_save_path, index=False)
                print('train_r2: %.7f train_rmse: %.7f train_LogS_1: %.7f train_LogS_point7: %.7f\nsolute_r2: %.7f solute_rmse: %.7f solute_LogS_1: %.7f solute_LogS_point7: %.7f\nsolvent_r2: %.7f solvent_rmse: %.7f solvent_LogS_1: %.7f solvent_LogS_point7: %.7f\ntemperature_r2: %.7f temperature_rmse: %.7f temperature_LogS_1: %.7f temperature_LogS_point7: %.7f\n' % (train_r2, train_rmse, train_1_percnetage, train_point7_percnetage, solute_r2, solute_rmse, solute_1_percnetage, solute_point7_percnetage, solvent_r2, solvent_rmse, solvent_1_percnetage, solvent_point7_percnetage, temperature_r2, temperature_rmse, temperature_1_percnetage, temperature_point7_percnetage))
                writer['train_rmse'].add_scalar('rmse', train_rmse, epoch + 1)
                writer['solute_rmse'].add_scalar('rmse', solute_rmse, epoch + 1)
                writer['solvent_rmse'].add_scalar('rmse', solvent_rmse, epoch + 1)
                writer['temperature_rmse'].add_scalar('rmse', temperature_rmse, epoch + 1)
                writer['train_r2'].add_scalar('r2', train_r2, epoch + 1)
                writer['solute_r2'].add_scalar('r2', solute_r2, epoch + 1)
                writer['solvent_r2'].add_scalar('r2', solvent_r2, epoch + 1)
                writer['temperature_r2'].add_scalar('r2', temperature_r2, epoch + 1)
                writer['train_log.7'].add_scalar('log.7', train_point7_percnetage, epoch + 1)
                writer['solute_log.7'].add_scalar('log.7', solute_point7_percnetage, epoch + 1)
                writer['solvent_log.7'].add_scalar('log.7', solvent_point7_percnetage, epoch + 1)
                writer['temperature_log.7'].add_scalar('log.7', temperature_point7_percnetage, epoch + 1)
                writer['train_log1'].add_scalar('log1', train_1_percnetage, epoch + 1)
                writer['solute_log1'].add_scalar('log1', solute_1_percnetage, epoch + 1)
                writer['solvent_log1'].add_scalar('log1', solvent_1_percnetage, epoch + 1)
                writer['temperature_log1'].add_scalar('log1', temperature_1_percnetage, epoch + 1)
                if all_rmse < best_rmse:
                    best_rmse = all_rmse
                    best_epoch = epoch + 1
                    if config['model_save']:
                        torch.save(model.state_dict(), save_path)
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
                lr_scheduler.step(solute_rmse)
            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)
            average_epoch_time = sum(epoch_times) / len(epoch_times)
            print(f'Epoch{epoch + 1:3d},Time:{epoch_time / 60:.2f} min,average_epoch_time:{average_epoch_time / 60:.2f} min\n')
        print('best_rmse:', best_rmse)
        print('best_epoch:', best_epoch)
        print('Training Finished!')
    if config['metric_type'] == 'cosmo_param':
        for epoch in range(epochs):
            start_time = time.time()
            all_predictions = []
            all_labels = []
            for batch in loader:
                labels = {'solvation_free_energy': batch['solvation_free_energy']}
                solute_ids = batch['Solute_SMILES_ids'].to(device)
                solvent_ids = batch['Solvent_SMILES_ids'].to(device)
                solute_mask = batch['Solute_attention_mask'].to(device)
                solvent_mask = batch['Solvent_attention_mask'].to(device)
                optimizer.zero_grad()
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    output = model(solute_ids, solvent_ids, solute_mask, solvent_mask)
                    loss = define_loss_function_2(config, output, labels, normalizers)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('max_grad_norm', 1.0))
                scaler.step(optimizer)
                scaler.update()
            with torch.no_grad():
                model.eval()
                for batch_val in COSMO_valid_loader:
                    labels_val = {'solvation_free_energy': batch_val['solvation_free_energy']}
                    solute_ids_val = batch_val['Solute_SMILES_ids'].to(device)
                    solvent_ids_val = batch_val['Solvent_SMILES_ids'].to(device)
                    solute_mask_val = batch_val['Solute_attention_mask'].to(device)
                    solvent_mask_val = batch_val['Solvent_attention_mask'].to(device)
                    output_val = model(solute_ids_val, solvent_ids_val, solute_mask_val, solvent_mask_val)
                    loss_val = define_loss_function_2(config, output_val, labels_val, normalizers)
                print('loss_train: %.7f loss_val: %.7f\n' % (loss, loss_val))
                writer['loss_train'].add_scalar('loss', loss, epoch + 1)
                writer['loss_val'].add_scalar('loss', loss_val, epoch + 1)
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_epoch = epoch + 1
                    if config['model_save']:
                        state_to_save = {'asym_cross_attention_block': model.asym_cross_attention_block.state_dict(), 'Thermodynamic_parameter_cal': model.Thermodynamic_parameter_cal.state_dict()}
                        torch.save(state_to_save, save_path_pretrain)
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
                    if early_stopping_count >= config['early_stop_patience']:
                        print(f'\nEarly stopping at epoch {epoch + 1}')
                        break
                lr_scheduler.step(loss_val)
            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)
            average_epoch_time = sum(epoch_times) / len(epoch_times)
            print(f'Epoch{epoch + 1:3d},Time:{epoch_time / 60:.2f} min,average_epoch_time:{average_epoch_time / 60:.2f} min\n')
        print('best_loss:', best_loss)
        print('best_epoch:', best_epoch)
        print('Training Finished!')
    if config['metric_type'] == 'inter_param':
        dataset_inter_train.prefetch_next = False
        if hasattr(dataset_inter_train, 'prefetch_executor') and dataset_inter_train.prefetch_executor:
            dataset_inter_train.prefetch_executor.shutdown(wait=True)
            dataset_inter_train.prefetch_executor = None
        print('Testing data loading...')
        test_start = time.time()
        try:
            test_batch = next(iter(loader))
            print(f'First batch loaded in {time.time() - test_start:.2f} seconds')
            print(f'Batch keys: {list(test_batch.keys())}')
            if 'Solute_SMILES_ids' in test_batch:
                print(f"Batch size: {test_batch['Solute_SMILES_ids'].shape}")
        except Exception as e:
            print(f'Error loading first batch: {e}')
            print('Trying alternative data loading approach...')
        for epoch in range(epochs):
            start_time = time.time()
            all_predictions = []
            all_labels = []
            print(loader)
            train_bar = tqdm(loader, desc=f'Epoch {epoch + 1}/{epochs} (Inter)', file=sys.stdout)
            for batch_idx, batch in enumerate(train_bar):
                labels = {'polarity_compatibility': batch['polarity_compatibility'], 'size_compatibility': batch['size_compatibility'], 'hbond_compatibility': batch['hbond_compatibility'], 'hydrophobicity_compatibility': batch['hydrophobicity_compatibility'], 'electrostatic_compatibility': batch['electrostatic_compatibility'], 'flexibility_compatibility': batch['flexibility_compatibility']}
                solute_ids = batch['Solute_SMILES_ids'].to(device)
                solvent_ids = batch['Solvent_SMILES_ids'].to(device)
                solute_mask = batch['Solute_attention_mask'].to(device)
                solvent_mask = batch['Solvent_attention_mask'].to(device)
                optimizer.zero_grad()
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    output = model(solute_ids, solvent_ids, solute_mask, solvent_mask)
                    loss = define_loss_function_2(config, output, labels, normalizers)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.get('max_grad_norm', 1.0))
                scaler.step(optimizer)
                scaler.update()
            with torch.no_grad():
                model.eval()
                for batch_val in inter_valid_loader:
                    labels_val = {'polarity_compatibility': batch_val['polarity_compatibility'], 'size_compatibility': batch_val['size_compatibility'], 'hbond_compatibility': batch_val['hbond_compatibility'], 'hydrophobicity_compatibility': batch_val['hydrophobicity_compatibility'], 'electrostatic_compatibility': batch_val['electrostatic_compatibility'], 'flexibility_compatibility': batch_val['flexibility_compatibility']}
                    solute_ids_val = batch_val['Solute_SMILES_ids'].to(device)
                    solvent_ids_val = batch_val['Solvent_SMILES_ids'].to(device)
                    solute_mask_val = batch_val['Solute_attention_mask'].to(device)
                    solvent_mask_val = batch_val['Solvent_attention_mask'].to(device)
                    output_val = model(solute_ids_val, solvent_ids_val, solute_mask_val, solvent_mask_val)
                    loss_val = define_loss_function_2(config, output_val, labels_val, normalizers)
                print('loss_train: %.7f loss_val: %.7f\n' % (loss, loss_val))
                writer['loss_train'].add_scalar('loss', loss, epoch + 1)
                writer['loss_val'].add_scalar('loss', loss_val, epoch + 1)
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_epoch = epoch + 1
                    if config['model_save']:
                        state_to_save = {'sym_cross_attention_block': model.sym_cross_attention_block.state_dict(), 'Interaction_parameter_cal': model.Interaction_parameter_cal.state_dict()}
                        torch.save(state_to_save, save_path_pretrain)
                    early_stopping_count = 0
                else:
                    early_stopping_count += 1
                    if early_stopping_count >= config['early_stop_patience']:
                        print(f'\nEarly stopping at epoch {epoch + 1}')
                        break
                lr_scheduler.step(loss_val)
            end_time = time.time()
            epoch_time = end_time - start_time
            epoch_times.append(epoch_time)
            average_epoch_time = sum(epoch_times) / len(epoch_times)
            print(f'Epoch{epoch + 1:3d},Time:{epoch_time / 60:.2f} min,average_epoch_time:{average_epoch_time / 60:.2f} min\n')
        print('best_loss:', best_loss)
        print('best_epoch:', best_epoch)
        print('Training Finished!')