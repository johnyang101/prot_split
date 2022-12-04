# %%
#Module imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

# %%
dataset_folder = "../google_prot_fns/"
filters = 64
epochs = 60
batch_size = 128
protein_len = 200

# %%
toy_dataset = pd.read_csv('/Mounts/rbg-storage1/users/johnyang/prot_split/data/2_class.csv')
classes = pd.unique(toy_dataset['family_accession'])

# %%
'''
Get 10 random proteins from each class
'''
minimal_dataset = toy_dataset.groupby('family_accession').apply(lambda x: x.sample(10))
minimal_dataset = minimal_dataset.reset_index(drop=True)

# %%
len(classes) #Should be 2

# %%
# from utils.gpu_utils import *

# %%
# chosen_gpu = get_free_gpu()
device = 'cuda'

# %%
# esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D") #Downloads cache to AFS. Limited space on AFS...
# batch_converter = alphabet.get_batch_converter()
# esm_model.to(device)
# esm_model.eval()
# print('done')

# %%
device = 'cuda'

# %%
class ESMFnDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, classes, device='cuda', max_len=200):
        self.dataset = dataset
        self.classes = classes
        self.max_len = max_len
        self.device = device
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        sequence_name = row['sequence_name']
        sequence_name = sequence_name.replace('/', '-')
        embeddings = torch.load(f'/Mounts/rbg-storage1/users/johnyang/prot_split/data/toy_esm_embeddings/{sequence_name}.pt')
        embeddings = torch.tensor(embeddings, device='cuda', dtype=torch.float32)

        '''Pad embeddings to max_len with zero vector'''
        if embeddings.size(1) < self.max_len:
            B, N, h = embeddings.size()
            pad = torch.zeros((B, self.max_len - embeddings.shape[1], h), device=self.device)
            embeddings = torch.cat((embeddings, pad), dim=1)

        class_idx = torch.tensor(self.class_to_idx[row['family_accession']])
        # label = F.one_hot(class_idx, num_classes=len(self.classes))
        embeddings = embeddings.squeeze(0) #NOTE: LS adds its own batch dimension, so we need to remove it here
        return embeddings, class_idx

class ESMFnMLPDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, classes, device='cuda', max_len=200):
        self.dataset = dataset
        self.classes = classes
        self.max_len = max_len
        self.device = device
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.idx_to_class = {i: classes[i] for i in range(len(classes))}
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        sequence_name = row['sequence_name']
        sequence_name = sequence_name.replace('/', '-')
        embeddings = torch.load(f'/Mounts/rbg-storage1/users/johnyang/prot_split/data/toy_esm_embeddings/{sequence_name}.pt')
        embeddings = torch.tensor(embeddings, device='cuda', dtype=torch.float32)

        '''Pad embeddings to max_len with zero vector'''
        if embeddings.size(1) < self.max_len:
            B, N, h = embeddings.size()
            pad = torch.zeros((B, self.max_len - embeddings.shape[1], h), device=self.device)
            embeddings = torch.cat((embeddings, pad), dim=1)

        class_idx = torch.tensor(self.class_to_idx[row['family_accession']])
        # label = F.one_hot(class_idx, num_classes=len(self.classes))
        embeddings = embeddings.squeeze(0) #NOTE: LS adds its own batch dimension, so we need to remove it here
        embeddings = embeddings.flatten()
        return embeddings, class_idx

# %%
dataset = ESMFnDataset(minimal_dataset, classes, device)
mlp_dataset = ESMFnMLPDataset(minimal_dataset, classes, device)


# %% [markdown]
# # Learning to Split

# %%
import ls

# %%
# python scripts/extract.py esm2_t33_650M_UR50D examples/data/some_proteins.fasta \
#   examples/data/some_proteins_emb_esm2 --repr_layers 33 --include per_tok

# %%
from ls.models.build import ModelFactory


# %%
@ModelFactory.register("esm_transformer")
class TransformerEncoder(torch.nn.Module):
    
    def __init__(self, include_label: int, input_size=1280, nheads=8, num_layers=6, device='cuda', num_classes=2, max_len=202, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.input_size = input_size
        self.device = device

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.input_size, nhead=nheads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        last_hidden_dim = 16
        self.fc1 = torch.nn.Linear(self.input_size, last_hidden_dim)

        self.include_label = include_label
        self.num_classes = num_classes
        self.output_fc = nn.Linear(last_hidden_dim * max_len + self.include_label, self.num_classes)

    def forward(self, embedding, y=None):
        if len(embedding.size()) > 3 and embedding.size(0) == 1:
            embedding = embedding.squeeze(0)
            assert len(embedding.size()) == 3, 'Embedding has greater than 4 dimensions'
            
        B, N, h = embedding.shape
        hidden = self.transformer_encoder(embedding)
        hidden = self.fc1(hidden)
        '''Flatten hidden state'''
        hidden = hidden.view(B, -1)
        if self.include_label > 0:
            # Append the one hot label embedding
            hidden = torch.cat(
                [hidden, F.one_hot(y, num_classes=self.include_label).float()],
                dim=1
            )
        hidden = self.output_fc(hidden)
        return hidden

# %%
@ModelFactory.register("esm_mlp")
class LinearLayer(torch.nn.Module):
    
    def __init__(self, include_label: int, input_size=1280, device='cuda', classes=2, max_len=202, **kwargs):
        super(LinearLayer, self).__init__()
        self.input_size = input_size
        self.classes = classes
        self.fc1 = torch.nn.Linear(self.input_size, 32)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32 * max_len, self.classes)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.device = device
        
    def forward(self, embedding):
        if len(embedding.size()) > 3 and embedding.size(0) == 1:
            embedding = embedding.squeeze(0)
            assert len(embedding.size()) == 3, 'Embedding has greater than 4 dimensions'
            
        B, N, h = embedding.shape
        # hidden = self.fc1(hidden)
        # hidden = self.fc2(hidden)
        '''Flatten hidden state'''
        hidden = self.fc1(embedding)
        hidden = self.relu1(hidden)
        hidden = hidden.view(B, -1)
        hidden = self.fc2(hidden)
        # hidden = self.softmax(hidden)
        return hidden

# %%
# %%
# data = ls.datasets.Tox21()

# # Learning to split the Tox21 dataset.
# # Here we use a simple mlp as our model backbone and use roc_auc as the evaluation metric.
# train_data, test_data, train_indices, test_indices, splitter = ls.learning_to_split(data, model={'name': 'mlp'}, metric='roc_auc', return_order=['train_data', 'test_data', 'train_indices', 'test_indices', 'splitter'], num_outer_loop=1)

# %%
# data.__getitem__(0)

# %%
# type(data)

# %%
dataset.__getitem__(0)[0].shape

# %%
# train_data, test_data, train_indices, test_indices, splitter = ls.learning_to_split(mlp_dataset, model={'name': 'mlp', 'args': {'hidden_dim_list': [1280 * 202, 256, 32]}}, 
#                                                                                     metric='accuracy', num_workers=0,
#                                                                                     return_order=['train_data', 'test_data', 'train_indices', 'test_indices', 'splitter'],
#                                                                                     batch_size=20, patience=1, num_outer_loop=2)

train_data, test_data, train_indices, test_indices, splitter = ls.learning_to_split(dataset, model={'name': 'esm_transformer', 'args': {'nheads': 1, 'num_layers': 2,}}, 
                                                                                    metric='accuracy', num_workers=0,
                                                                                    return_order=['train_data', 'test_data', 'train_indices', 'test_indices', 'splitter'],
                                                                                    batch_size=20, patience=1, num_outer_loop=2)