import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch_geometric
import gvpdir
import os
import re
import esm
from esm.model.esm2 import ESM2
from utils import pd_utils as pdu
from utils import data_utils as dadu
from scipy.stats import spearmanr
import utils.data_utils
from Bio.PDB.DSSP import DSSP
from Bio.PDB.MMCIFParser import MMCIFParser

plt.rcParams['figure.facecolor'] = 'white'

def get_residue_positions(ah, bs, ex_seq):
    for i, r in enumerate(ah + bs):
        s, e = r
        for j in range(s, e + 1, 1):
            char = ex_seq[j]
            if i < len(ah):
                if char == 'V':
                    print(f'alpha helix V position {j}')
                elif char == 'A':
                    print(f'alpha helix A position {j}')
                elif char == 'K':
                    print(f'alpha helix K position {j}')
            else:
                if char == 'V':
                    print(f'bs V position {j}')
                elif char == 'A':
                    print(f'bs A position {j}')
                elif char == 'K':
                    print(f'bs K position {j}')
                elif char == 'Y':
                    print(f'bs Y position {j}')

def ind_to_letter(batch):
    letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                'N': 2, 'Y': 18, 'M': 12}
    num_to_letter = {v:k for k, v in letter_to_num.items()}

    seq = [num_to_letter[x.item()] for x in batch.seq[batch.mask].cpu()]

    output = ''.join(seq)
    return output

def generate_new_train_seq(max_nodes_s, dataloader):
    num_workers_s = 41

    for batch in dataloader:
        if len(batch.name) == 1:
            toy = batch
            print('hit')
            break

    output = ind_to_letter(toy)
    print(output)
    return output, toy
    
def get_codebook_vectors(positions, batch, model, esm_flag=False):
    model.eval()
    codebook_vectors = {}

    with torch.no_grad():
        logits, commit_loss, embed_idx = dadu.call_model(model, batch, esm_flag=esm_flag)
        embed_idx = embed_idx.squeeze(0)
        for pos in positions:
            if not batch.mask[pos]:
                continue
            code = embed_idx[pos]
            codebook_vectors[pos] = model.node_s_vq.codebook[code, :]
    return codebook_vectors

'''Compute pairwise cosine similarities between codebook vectors in a dictionary and store in a dictionary'''
def pairwise_cos_sim_dict(pos_codebook_vecs):
    cos_sim_dict = {}
    for pos1 in pos_codebook_vecs:
        cos_sim_dict[pos1] = {}
        for pos2 in pos_codebook_vecs:
            if pos1 == pos2:
                continue
            cos_sim = torch.nn.functional.cosine_similarity(pos_codebook_vecs[pos1], pos_codebook_vecs[pos2], dim=0)
            cos_sim_dict[pos1][pos2] = cos_sim.item()
    return cos_sim_dict
    
def calc_residue_dist(residue_one, residue_two) :
    """Returns the C-alpha distance between two residues"""
    diff_vector  = residue_one["CA"].coord - residue_two["CA"].coord
    return np.sqrt(np.sum(diff_vector * diff_vector))

def calc_dist_matrix(chain_one, chain_two) :
    """Returns a matrix of C-alpha distances between two chains"""
    chain_one_residues = [r for r in chain_one.get_residues() if r.get_id()[0] == " "]
    chain_two_residues = [r for r in chain_two.get_residues() if r.get_id()[0] == " "]
    answer = np.zeros((len(chain_one_residues), len(chain_two_residues)), np.float64)
    for i, residue_one in enumerate(chain_one_residues) :
        for j, residue_two in enumerate(chain_two_residues) :
            answer[i, j] = calc_residue_dist(residue_one, residue_two)
    return answer

'''Create distance matrix of latent space vectors'''
def create_dist_matrix(codebook_vectors, positions): #TODO: Reformat this code to compensate for other data structures.
    dist_matrix = np.zeros((len(positions), len(positions)))
    for i, pos1 in enumerate(positions):
        for j, pos2 in enumerate(positions):
            dist_matrix[i, j] = torch.nn.functional.cosine_similarity(codebook_vectors[pos1], codebook_vectors[pos2], dim=0)
    return dist_matrix

def _load_model_and_alphabet_core_v2(model_data):
    def upgrade_state_dict(state_dict):
        """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
        prefixes = ["encoder.sentence_encoder.", "encoder."]
        pattern = re.compile("^" + "|".join(prefixes))
        state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
        return state_dict

    cfg = model_data["cfg"]["model"]
    state_dict = model_data["model"]
    state_dict = upgrade_state_dict(state_dict)
    alphabet = esm.data.Alphabet.from_architecture("ESM-1b")
    model = ESM2(
        num_layers=cfg.encoder_layers,
        embed_dim=cfg.encoder_embed_dim,
        attention_heads=cfg.encoder_attention_heads,
        alphabet=alphabet,
        token_dropout=cfg.token_dropout,
    )
    return model, alphabet, state_dict

def _load_model_and_alphabet_core_v1(model_data):
    import esm  # since esm.inverse_folding is imported below, you actually have to re-import esm here

    alphabet = esm.Alphabet.from_architecture(model_data["args"].arch)

    if model_data["args"].arch == "roberta_large":
        # upgrade state dict
        pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
        prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
        prs2 = lambda s: "".join(
            s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
        )
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs1(prs2(arg[0])): arg[1] for arg in model_data["model"].items()}
        model_state["embed_tokens.weight"][alphabet.mask_idx].zero_()  # For token drop
        model_args["emb_layer_norm_before"] = has_emb_layer_norm_before(model_state)
        model_type = esm.ProteinBertModel

    elif model_data["args"].arch == "protein_bert_base":

        # upgrade state dict
        pra = lambda s: "".join(s.split("decoder_")[1:] if "decoder" in s else s)
        prs = lambda s: "".join(s.split("decoder.")[1:] if "decoder" in s else s)
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs(arg[0]): arg[1] for arg in model_data["model"].items()}
        model_type = esm.ProteinBertModel
    elif model_data["args"].arch == "msa_transformer":

        # upgrade state dict
        pra = lambda s: "".join(s.split("encoder_")[1:] if "encoder" in s else s)
        prs1 = lambda s: "".join(s.split("encoder.")[1:] if "encoder" in s else s)
        prs2 = lambda s: "".join(
            s.split("sentence_encoder.")[1:] if "sentence_encoder" in s else s
        )
        prs3 = lambda s: s.replace("row", "column") if "row" in s else s.replace("column", "row")
        model_args = {pra(arg[0]): arg[1] for arg in vars(model_data["args"]).items()}
        model_state = {prs1(prs2(prs3(arg[0]))): arg[1] for arg in model_data["model"].items()}
        if model_args.get("embed_positions_msa", False):
            emb_dim = model_state["msa_position_embedding"].size(-1)
            model_args["embed_positions_msa_dim"] = emb_dim  # initial release, bug: emb_dim==1

        model_type = esm.MSATransformer

    elif "invariant_gvp" in model_data["args"].arch:
        import esm.inverse_folding

        model_type = esm.inverse_folding.gvp_transformer.GVPTransformerModel
        model_args = vars(model_data["args"])  # convert Namespace -> dict

        def update_name(s):
            # Map the module names in checkpoints trained with internal code to
            # the updated module names in open source code
            s = s.replace("W_v", "embed_graph.embed_node")
            s = s.replace("W_e", "embed_graph.embed_edge")
            s = s.replace("embed_scores.0", "embed_confidence")
            s = s.replace("embed_score.", "embed_graph.embed_confidence.")
            s = s.replace("seq_logits_projection.", "")
            s = s.replace("embed_ingraham_features", "embed_dihedrals")
            s = s.replace("embed_gvp_in_local_frame.0", "embed_gvp_output")
            s = s.replace("embed_features_in_local_frame.0", "embed_gvp_input_features")
            return s

        model_state = {
            update_name(sname): svalue
            for sname, svalue in model_data["model"].items()
            if "version" not in sname
        }

    else:
        raise ValueError("Unknown architecture selected")

    model = model_type(
        Namespace(**model_args),
        alphabet,
    )

    return model, alphabet, model_state

def load_model_and_alphabet_core(model_name, model_data, regression_data=None):
    if regression_data is not None:
        model_data["model"].update(regression_data["model"])

    if model_name.startswith("esm2"):
        model, alphabet, model_state = _load_model_and_alphabet_core_v2(model_data)
    else:
        model, alphabet, model_state = _load_model_and_alphabet_core_v1(model_data)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    if regression_data is None:
        expected_missing = {"contact_head.regression.weight", "contact_head.regression.bias"}
        error_msgs = []
        missing = (expected_keys - found_keys) - expected_missing
        if missing:
            error_msgs.append(f"Missing key(s) in state_dict: {missing}.")
        unexpected = found_keys - expected_keys
        if unexpected:
            error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected}.")

        if error_msgs:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    model.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        if expected_missing - found_keys:
            warnings.warn(
                "Regression weights not found, predicting contacts will not produce correct results."
            )

    model.load_state_dict(model_state, strict=regression_data is not None)

    return model, alphabet

def get_token_representations(seq, model, alphabet, batch_converter, device):
    data = [
        ("protein1", f"{seq}"),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    batch_tokens = batch_tokens.to(device)

    # Extract per-residue representations (on CPU) #TODO: Change this to cuda somehow.
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    token_representations = results["representations"][33]
    return token_representations

def esm_pairwise_cosine_similarity_matrix(seq, esm):
    '''Create cosine similarity matrix of token representations'''
    esm = esm.cpu()
    esm = esm / esm.norm(dim=-1, keepdim=True)
    token_similarity = torch.einsum("bld,brd->blr", esm, esm)
    distances = token_similarity.squeeze(0).numpy()[: len(seq), : len(seq)]
    print(f'ESM CS distance matrix is {distances.shape}\n')
    return distances

def ee_pairwise_cosine_similarity_matrix(batch, device='cuda'):
    '''Create cosine similarity matrix of encoder embeddings'''
    node_dim = (100, 16)
    edge_dim = (32, 1)
    model = gvpdir.gvp.models.OldCPDModel((6, 3), node_dim, (32, 1), edge_dim).to(device)
    sd = torch.load('/afs/csail.mit.edu/u/j/johnyang/home/s2sDM/gvpdir/models/1658695209_99.pt')
    model.load_state_dict(sd)

    batch = batch.to(device)
    h_V = (batch.node_s, batch.node_v)
    h_E = (batch.edge_s, batch.edge_v)
    ee = model.encoder_embeddings(h_V, batch.edge_index, h_E, seq=batch.seq)
    ee = ee[0][batch.mask].cpu()
    ee = ee / ee.norm(dim=-1, keepdim=True)
    ee_similarity = torch.einsum("ld,rd->lr", ee, ee)
    bseq = batch.seq[batch.mask]
    ee_distances = ee_similarity.squeeze(0).detach().numpy()[: len(bseq), : len(bseq)]
    print(f'EE CS distance matrix is {ee_distances.shape}\n')
    return ee_distances

def matrices_spearmanr(matrices, names):
    '''Get spearman correlations between all pairwise combinations of matrices'''
    from scipy.stats import spearmanr
    arr = [[spearmanr(matrices[i].flatten(), matrices[j].flatten())[0] for j in range(len(matrices))] for i in range(len(matrices))]
    df = pd.DataFrame(arr, columns=names, index=names)
    return df

'''Order the codebook vectors by their distance to the query vector'''
def order_codebook_vectors(query_vector, codebook):
    distances = torch.einsum("ld,rd->lr", query_vector, codebook)
    print(distances)
    distances = distances.squeeze(0).detach()
    ind = torch.argsort(distances, descending=True)
    return codebook[ind], [x.item() for x in ind]

def ss_spearmanr(batch, codebook, embed_idx):
    mmcifpath = utils.data_utils.get_jyim_mmcif_path(batch.name[0])
    p = MMCIFParser()
    structure = p.get_structure(batch.name[0], mmcifpath)
    model = structure[0]
    dssp = DSSP(model, mmcifpath)
    simple_dssp_map = {
        'H': 0,
        'B': 1,
        'E': 1,
        'G': 0,
        'I': 0,
        'T': 2,
        'S': 2,
        ' ': 2,
        '-': 2
    }
    pos_ss = [simple_dssp_map[dssp[('B', (' ', i, ' '))][2]] for i in range(len(batch.seq)) if ('B', (' ', i, ' ')) in dssp.keys() and batch.mask[i].item()]
    _, sort_order = order_codebook_vectors(codebook[0], codebook) #TODO: Select a helix vector
    relative_ranking = [sort_order.index(embed_idx[i]) for i in range(len(batch.seq)) if ('B', (' ', i, ' ')) in dssp.keys() and batch.mask[i].item()]
    return spearmanr(pos_ss, relative_ranking)

def get_ckpt_model(ckpt_dir):
    ckpt_files = [
        x for x in os.listdir(ckpt_dir)
        if 'pkl' in x or '.pth' in x
    ]
    if len(ckpt_files) != 1:
        raise ValueError(f'Ambiguous ckpt in {ckpt_dir}')
    ckpt_name = ckpt_files[0]
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    ckpt_pkl = pdu.read_pkl(ckpt_path, use_torch=True)
    ckpt_model = ckpt_pkl['model_state_dict']
    return ckpt_model

def s_vs_s_context_table(cb_dict):
    v = cb_dict[55]
    rows = ['Similar Structure', "Different Structure"]
    cols = ['Similar Residue', "Different Residue"]
    assert type(v[46]) is not torch.tensor, 'Does not accept torch tensors.'
    arr = [[v[46], v[33]], [v[99], v[16]]]
    df = pd.DataFrame(arr)
    df.index = rows
    df.columns = cols
    return df

def ls_heatmap(cb_dict, save_path=None):
    positions = [33, 46, 55, 15, 16, 17, 99, 84, 31]
    names = ['K-A pos 34', 'A-A pos 47', 'V-A pos 56', 'Y-B pos 16', 'K-B pos 17', 'P-B pos 18', 'V-B pos 100', 'V-L pos 85', 'C-A,disulfide pos 32']
    for k, v in cb_dict.items(): v[k] = 1.0    
    arr = [[cb_dict[i][j] for j in positions] for i in positions]
    cs_df = pd.DataFrame(arr)
    cs_df.index = names
    cs_df.columns = names
    with plt.rc_context({'figure.figsize': (12, 12)}):
        sns.heatmap(cs_df, vmin=0, vmax=1)
        if save_path:
            plt.savefig(save_path)
        plt.show()
    return cs_df