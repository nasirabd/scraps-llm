import torch

def _extract_ids(sample):
    """
    Accepts:
      - list[int] / tuple[int] / np.array
      - dict with key 'ids'
      - your EncodeOut (has .input_ids)
      - HF tokenizers Encoding (has .ids)
    Returns list[int].
    """
    if isinstance(sample, dict) and "ids" in sample:
        sample = sample["ids"]
    if hasattr(sample, "input_ids"): 
        sample = sample.input_ids
    elif hasattr(sample, "ids"):      
        sample = sample.ids
    return list(sample)

def pad_collate(batch, pad_id: int):
    """
    Collate function for variable-length tokenized sequences.

    Args:
        batch: list of samples (various formats supported, see _extract_ids)
        pad_id: int, padding token id

    Returns:
        x: (B, T-1) LongTensor inputs
        y: (B, T-1) LongTensor targets (PAD masked as -100)
    """
    if not batch:
        return (torch.empty(0, 0, dtype=torch.long),
                torch.empty(0, 0, dtype=torch.long))

    # Normalize all items to list[int]
    seqs = [_extract_ids(s) for s in batch]

    # Ensure each sequence is at least length 2 so shifting doesn't underflow
    seqs = [s if len(s) >= 2 else (s + [pad_id]) for s in seqs]

    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)

    for i, ids in enumerate(seqs):
        L = len(ids)
        out[i, :L] = torch.tensor(ids, dtype=torch.long)

    # Autoregressive shift
    x = out[:, :-1]
    y = out[:, 1:].clone()

    # Mask padding in the loss
    y[out[:, 1:] == pad_id] = -100

    return x, y

