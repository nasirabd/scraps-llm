# src/tokenization/bpe_tok.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

from tokenizers import Tokenizer

SPECIALS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "<ING>", "<REC>", "<TITLE>", "<STEP>"]

@dataclass
class EncodeOut:
    input_ids: List[int]
    attention_mask: List[int]

class BPETok:
    def __init__(self, path: str = "tokenizer/bpe.json"):
        self.tk: Tokenizer = Tokenizer.from_file(path)

        # Resolve IDs (guard against None)
        self.pad_id  = self._tok_id_or_raise("[PAD]")
        self.bos_id  = self._tok_id_or_raise("[BOS]")
        self.eos_id  = self._tok_id_or_raise("[EOS]")
        self.unk_id  = self._tok_id_or_raise("[UNK]")

        # Section tokens (optional: allow missing)
        self.ing_id   = self._tok_id_or_none("<ING>")
        self.rec_id   = self._tok_id_or_none("<REC>")
        self.title_id = self._tok_id_or_none("<TITLE>")
        self.step_id  = self._tok_id_or_none("<STEP>")

        # Detect if a post-processor already inserts BOS/EOS
        # (HuggingFace Tokenizers can do this; we just check by a tiny probe)
        probe = self.tk.encode("x").ids
        self.inserts_bos_eos = (len(probe) >= 2 and (probe[0] == self.bos_id or probe[-1] == self.eos_id))

    def _tok_id_or_none(self, tok: str) -> Optional[int]:
        tid = self.tk.token_to_id(tok)
        return tid if tid is not None else None

    def _tok_id_or_raise(self, tok: str) -> int:
        tid = self.tk.token_to_id(tok)
        if tid is None:
            raise ValueError(f"Special token {tok!r} not found in tokenizer vocab. "
                             "Make sure it was included in the trainer's special_tokens.")
        return tid

    # -------- single-item encode/decode --------
    def encode(self, text: str, add_special: bool = True, max_len: Optional[int] = None) -> EncodeOut:
        ids = self.tk.encode(text).ids

        # Avoid double-add if post-processor already did it
        if add_special and not self.inserts_bos_eos:
            ids = [self.bos_id] + ids + [self.eos_id]

        if max_len is not None:
            ids = ids[:max_len]

        attn = [1] * len(ids)
        return EncodeOut(ids, attn)

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> str:
        if stop_at_eos and self.eos_id in ids:
            ids = ids[:ids.index(self.eos_id)]
        # Skip special tokens during detok to avoid artifacts
        return self.tk.decode(list(ids), skip_special_tokens=True)

    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    # -------- batching utilities for training --------
    def pad_batch(
        self,
        batch_ids: List[List[int]],
        pad_to: Optional[int] = None,
        pad_on_left: bool = False,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        """
        Pad a list of ID lists to a common length. Returns (input_ids, attention_mask).
        """
        if pad_to is None:
            pad_to = max(len(x) for x in batch_ids) if batch_ids else 0

        padded_ids: List[List[int]] = []
        masks: List[List[int]] = []
        for ids in batch_ids:
            pad_len = max(0, pad_to - len(ids))
            pad_tokens = [self.pad_id] * pad_len
            pad_mask = [0] * pad_len
            if pad_on_left:
                padded_ids.append(pad_tokens + ids)
                masks.append(pad_mask + [1] * len(ids))
            else:
                padded_ids.append(ids + pad_tokens)
                masks.append([1] * len(ids) + pad_mask)
        return padded_ids, masks

    def encode_batch(
        self,
        texts: Sequence[str],
        add_special: bool = True,
        max_len: Optional[int] = None,
        pad_to: Optional[int] = None,
        pad_on_left: bool = False,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        outs = [self.encode(t, add_special=add_special, max_len=max_len) for t in texts]
        ids = [o.input_ids for o in outs]
        return self.pad_batch(ids, pad_to=pad_to, pad_on_left=pad_on_left)
