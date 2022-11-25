import numpy as np
from Bio.PDB import *
import sklearn
from sklearn.preprocessing import LabelBinarizer

def PDB_to_one_hot_seq(structure_id, file_path):

    def get_structure(structure_id, file_path):
        parser = PDBParser(PERMISSIVE=1, QUIET=True)
        ppb = PPBuilder()
        structure = parser.get_structure(structure_id, file_path)
    
        return structure
    
    def get_pp_sequences(structure):
        ppb = PPBuilder()
        return [str(pp.get_sequence()) for pp in ppb.build_peptides(structure)]
    
    def seq_to_one_hot_arr(sequence):
        '''
        @param: sequence - string of single-letter AAs

        output: one-hot encoding of sequence where ROW VECTORS correspond to residue one-
        hot vectors.
        '''

        d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
        aastr = ''.join(d.values())
        #aa_letter_to_index = {(aastr[i], i) for i in range(len(aastr))}

        print('fuck you')s
        label_binarizer = LabelBinarizer()
        label_binarizer.fit([char for char in aastr])

        b = label_binarizer.transform([char for char in sequence])
        assert b.shape == (len(sequence), 20), 'output shape not (len(sequence), 20)'
        return torch.tensor(b).float()
    
    structure = get_structure(structure_id, file_path)
    
    sequence = get_pp_sequences(structure)[0]
    
    return seq_to_one_hot_arr(sequence)


    
    