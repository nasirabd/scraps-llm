import torch

def pad_collate(batch, pad_id: int):
    """
    Collate function for variable-length tokenized sequences.

    Args:
        batch: list of samples, each a list[int] (token IDs)
        pad_id: int, ID used for padding

    Returns:
        x: (B, T-1) LongTensor of inputs
        y: (B, T-1) LongTensor of targets (with PAD masked as -100)
    """
    maxlen = max(len(x) for x in batch)        # longest sequence
    out = torch.full((len(batch), maxlen), pad_id, dtype=torch.long)

    for i, ids in enumerate(batch):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)

    # Shift for autoregressive LM
    x = out[:, :-1]
    y = out[:, 1:].clone()
    y[out[:, 1:] == pad_id] = -100   # mask pads in the loss

    return x, y
