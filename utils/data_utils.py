from gvpdir.gvp.models import CPD_VQ_MODEL, VQ_AE_MODEL, VQ_AE_FT_MODEL, CPD_VQ_ESM_MODEL, CPD_RVQ_MODEL
import torch_geometric
import re
import gvpdir
import torch
import os
import copy
import tqdm
import json
from gvpdir.gvp.data_ESM import ProteinGraphDataset
from gvpdir.gvp.models import OldCPDModel

def get_model(conf):
    if conf.model_type == 'CPD_VQ_MODEL': 
        model = CPD_VQ_MODEL(**conf)
    elif conf.model_type == 'VQ_AE_MODEL':
        model = VQ_AE_MODEL(**conf)
    elif conf.model_type == 'VQ_AE_FT_MODEL':
        model = VQ_AE_FT_MODEL(**conf)
    elif conf.model_type == 'CPD_VQ_ESM_MODEL':
        model = CPD_VQ_ESM_MODEL(**conf)
    elif conf.model_type == 'CPD_RVQ_MODEL':
        model = CPD_RVQ_MODEL(**conf)
    else:
        raise ValueError(f'Model type {conf.model_type} not supported.')
    return model

def get_dataloader(conf):
    esm_dl = True if re.search('ESM', conf.model.model_type) else False

    if esm_dl or conf.augment_eps > 0:
        dataloader = lambda x: torch_geometric.loader.DataLoader(x, 
            num_workers=conf.num_workers,
            batch_sampler=gvpdir.gvp.data_ESM.BatchSampler(
                x.node_counts, max_nodes=conf.batch_tokens))
    else:
        dataloader = lambda x: torch_geometric.loader.DataLoader(x, 
            num_workers=conf.num_workers,
            batch_sampler=gvpdir.gvp.data.BatchSampler(
                x.node_counts, max_nodes=conf.batch_tokens))

    trainset = torch.load(conf.trainset_path) 
    valset = torch.load(conf.valset_path)

    trainset.augment_eps = conf.augment_eps 

    trainloader = dataloader(trainset)
    valloader = dataloader(valset)

    return trainloader, valloader

def get_gvp_databatch(name, esm_flag):
    home_path = '/afs/csail.mit.edu/u/j/johnyang/home'
    cath_path = os.path.join(home_path, 'neurips19-graph-protein-design/data/cath')
    path = os.path.join(cath_path, "chain_set.jsonl")
    with open(path) as f:
        lines = f.readlines()
    
    for line in tqdm.tqdm(lines):
        entry = json.loads(line)
        if entry['name'] != name:
            continue
        print(f'Found {name} in dataset')
        coords = entry['coords']
        entry['coords'] = list(zip(
            coords['N'], coords['CA'], coords['C'], coords['O']
        ))
        if esm_flag:
            esm_chain_path = f'/Mounts/rbg-storage1/users/johnyang/saved_models/CATH_ESM_EMBEDDINGS/{name}.pt'
            entry['ESM'] = torch.load(esm_chain_path)["representations"][33]

        return ProteinGraphDataset([entry])._featurize_as_graph(entry)
    raise ValueError(f'Could not find {name} in dataset')

def call_model(model, obj, esm_flag=False):
    """Call model with obj and return loss."""
    h_V = (obj.node_s, obj.node_v)
    h_E = (obj.edge_s, obj.edge_v)
    if esm_flag:
        logits, commit_loss, embed_idx = model(h_V, obj.edge_index, h_E, seq=obj.seq, esm=obj.esm)
    else:
        logits, commit_loss, embed_idx = model(h_V, obj.edge_index, h_E, seq=obj.seq) #TODO: refactor models for backward compatibility
    return logits, commit_loss, embed_idx

def get_jyim_mmcif_path(name):
    MMCIF_DIR = '/data/rsg/chemistry/jyim/large_data/pdb/30_08_2021/mmCIF'
    pdb_name, cath_chain_id = name.split('.')
    pdb_subdir = pdb_name[1:3]
    mmcif_subdir = os.path.join(MMCIF_DIR, pdb_subdir)
    mmcif_path = os.path.join(mmcif_subdir, pdb_name + '.cif')
    return mmcif_path
       
def get_pretrained_gvp():
    model = OldCPDModel((6, 3), (100, 16), (32, 1), (32, 1))
    pretrained_sd = torch.load('/afs/csail.mit.edu/u/j/johnyang/home/s2sDM/gvpdir/models/1658695209_99.pt')
    model_sd = copy.deepcopy(model.state_dict())

    for param in model_sd.keys():
        if param in pretrained_sd:
            model_sd[param] = pretrained_sd[param]
    model.load_state_dict(model_sd)
    return model

def load_sd_for_matched_keys(model, sd):
    model_sd = copy.deepcopy(model.state_dict())
    for param in model_sd.keys():
        if param in sd:
            model_sd[param] = sd[param]
    model.load_state_dict(model_sd)
    return model