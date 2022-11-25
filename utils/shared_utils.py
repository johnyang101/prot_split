import torch

def create_mask(size, seq_ends, padded_false=True, device='cuda'):

    B, N = size
    mask_for_seq_list = []
    for b in range(B):
        end = seq_ends[b]
        if padded_false:
            mask_for_seq_list.append(torch.concat((torch.ones(end, dtype=torch.bool).to(device), torch.zeros(N-end, dtype=torch.bool).to(device))))
        else:
            mask_for_seq_list.append(torch.concat((torch.zeros(end, dtype=torch.bool).to(device), torch.ones(N-end, dtype=torch.bool).to(device))))


    mask = torch.stack(mask_for_seq_list, dim=0).to(device)
    assert(mask.size() == (B, N), 'mask size is wrong')
    return mask



    
    