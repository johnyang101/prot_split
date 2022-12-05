# %%
#Module imports
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
import ls # Yujia Bao Learning to Split
from ls.models.build import ModelFactory
import itertools
import os
from typing import Sequence, Tuple, List, Union
import pickle
import re
import shutil
import torch
import pathlib

# %%
class ESMFnDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, classes, data_path, device='cuda', max_len=200):
        self.dataset = dataset
        self.classes = classes
        self.data_path = data_path
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
        
        embeddings = torch.load(f'{self.data_path}/{sequence_name}.pt')
        embeddings = embeddings['representations'][33].to('cuda').float() # ESM Embedding output format, 33 is last layer of 
        # embeddings = torch.tensor(embeddings, device='cuda', dtype=torch.float32)
        '''Pad embeddings to max_len with zero vector'''
        if embeddings.size(1) < self.max_len:
            B, N, h = embeddings.size()
            pad = torch.zeros((B, self.max_len - embeddings.shape[1], h), device=self.device)
            embeddings = torch.cat((embeddings, pad), dim=1)

        class_idx = torch.tensor(self.class_to_idx[row['family_accession']])
        # label = F.one_hot(class_idx, num_classes=len(self.classes))
        embeddings = embeddings.squeeze(0) #NOTE: LS adds its own batch dimension, so we need to remove it here
        return embeddings, class_idx
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

RawMSA = Sequence[Tuple[str, str]]
class FastaBatchedDataset(object):
    def __init__(self, sequence_labels, sequence_strs):
        self.sequence_labels = list(sequence_labels)
        self.sequence_strs = list(sequence_strs)
    @classmethod
    def from_file(cls, fasta_file):
        sequence_labels, sequence_strs = [], []
        cur_seq_label = None
        buf = []
        def _flush_current_seq():
            nonlocal cur_seq_label, buf
            if cur_seq_label is None:
                return
            sequence_labels.append(cur_seq_label)
            sequence_strs.append("".join(buf))
            cur_seq_label = None
            buf = []
        with open(fasta_file, "r") as infile:
            for line_idx, line in enumerate(infile):
                if line.startswith(">"):  # label line
                    _flush_current_seq()
                    line = line[1:].strip()
                    if len(line) > 0:
                        cur_seq_label = line
                    else:
                        cur_seq_label = f"seqnum{line_idx:09d}"
                else:  # sequence line
                    buf.append(line.strip())
        _flush_current_seq()
        assert len(set(sequence_labels)) == len(
            sequence_labels
        ), "Found duplicate sequence labels"
        return cls(sequence_labels, sequence_strs)
    def __len__(self):
        return len(self.sequence_labels)
    def __getitem__(self, idx):
        return self.sequence_labels[idx], self.sequence_strs[idx]
    def get_batch_indices(self, toks_per_batch, extra_toks_per_seq=0):
        sizes = [(len(s), i) for i, s in enumerate(self.sequence_strs)]
        sizes.sort()
        batches = []
        buf = []
        max_len = 0
        def _flush_current_buf():
            nonlocal max_len, buf
            if len(buf) == 0:
                return
            batches.append(buf)
            buf = []
            max_len = 0
        for sz, i in sizes:
            sz += extra_toks_per_seq
            if max(sz, max_len) * (len(buf) + 1) > toks_per_batch:
                _flush_current_buf()
            max_len = max(max_len, sz)
            buf.append(i)
        _flush_current_buf()
        return batches

def get_data(fasta_file):
    esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    batch_converter = alphabet.get_batch_converter()
    esm_model.to('cuda')
    esm_model.eval()
    model = esm_model
    model.eval()

    toks_per_batch = 10000
    output_dir = '../data/results'
    output_file = 'run1'

    model = model.cuda()
    print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file('./selected_fams.fasta')
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")

    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = False

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in [33]) #[33] is the last layer of the thing
    repr_layers = [(i + model.num_layers + 1) % (model.num_layers + 1) for i in [33]]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                output_file = output_dir / f"{label}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                # assert os.path.is_dir(args.output_dir), f"Output directory {args.output_dir} does not exist"
                result = {"label": label}
                truncate_len = min(200, len(strs[i])) #TODO: Make that better, should be protien_len
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                
                if (1): # "per_tok" in args.include:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }
                torch.save(
                    result,
                    output_file,
                )
            del toks
            torch.cuda.empty_cache()

def return_from_dataset(dataset, classes):
  return [dataset.loc[dataset['family_accession'].isin(classes)].reset_index(), classes]

if __name__ == "__main__":
    dataset_folder = "../data/results"
    protein_len = 200
    # get_data('selected_fams.fasta')
    toy_dataset = pd.read_csv('../data/2_class.csv')
    classes = pd.unique(toy_dataset['family_accession'])

    full_data_temp = []
    for name_sub_folder in ["train", "dev", "test"]:
        for f in os.listdir(os.path.join("../google_prot_fns/", name_sub_folder)):
            data = pd.read_csv(os.path.join("../google_prot_fns/", name_sub_folder, f))
            full_data_temp.append(data)
        full_data_temp.append(pd.concat(full_data_temp))
    all_datasets = pd.concat([full_data_temp[0], full_data_temp[1], full_data_temp[2]])
    del full_data_temp

    sel_fams = np.load('../data/selected_fams.npy') 
    sel_dataset, _ = return_from_dataset(all_datasets, sel_fams)
    device = 'cuda'

    dataset = ESMFnDataset(sel_dataset, sel_fams, dataset_folder, device = device)
    num_classes = 63 #Get from the selected dataset, do not hardcode
    # torch.multiprocessing.set_start_method('spawn')
    train_data, test_data, train_indices, test_indices, splitter = ls.learning_to_split(dataset, model={'name': 'esm_transformer', 'args': {'nheads': 8, 'num_layers': 6, 'max_len': 200}}, 
                                                                                        metric='accuracy', return_order=['train_data', 'test_data', 'train_indices', 'test_indices', 'splitter'],
                                                                                        batch_size=20, num_workers=0, num_batches = 1050, patience=5, num_classes=num_classes) # TODO: Fix torch multiprocessing error for num_workers