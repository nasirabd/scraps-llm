from tokenizers import Tokenizer

class BPETok:
    def __init__(self, path="tokenizer/bpe.json"):
        # Load the trained tokenizer
        self.tk = Tokenizer.from_file(path)

        # Map special tokens to IDs
        self.pad_id = self.tk.token_to_id("[PAD]")
        self.bos_id = self.tk.token_to_id("[BOS]")
        self.eos_id = self.tk.token_to_id("[EOS]")
        self.unk_id = self.tk.token_to_id("[UNK]")
        # Section tokens
        self.ing_id = self.tk.token_to_id("<ING>")
        self.rec_id = self.tk.token_to_id("<REC>")
        self.title_id = self.tk.token_to_id("<TITLE>")
        self.step_id = self.tk.token_to_id("<STEP>")
    
    def encode(self,text: str, add_special=True):
        """Convert text to token IDs. Optionally add [BOS] ... [EOS]."""
        ids = self.tk.encode(text).ids
        if add_special:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self,ids):
        """Convert token IDs back into text, stopping at EOS if present."""
        if self.eos_id in ids:
            ids = ids[:ids.index(self.eos_id)]
        return self.tk.decode(ids)

    def vocab_size(self):
        return self.tk.get_vocab_size()