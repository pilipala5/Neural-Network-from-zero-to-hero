import regex as re
from base import Tokenizer, get_stats, merge


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RegexTokenizer(Tokenizer):
    """
    添加正则化和特殊令牌的分词器

    """

    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(pattern=self.pattern)
        self.special_tokens = {}    # str -> int, example: {'<|endoftext|>': 100257}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        ids_chunks = [list(ck.encode("utf-8", errors="replace") for ck in self.compiled_pattern.findall(text))]

        vocab_size = 256 + 3
        num_merges = vocab_size - 256
        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}

        idx = 256

        for i in range(num_merges):
            new_id = idx + i
            stats = {}
            for chunk in ids_chunks:
                if len(chunk) >= 2:
                    stats = get_stats(chunk, stats)

            pair = max(stats, key=stats.get)
            ids_chunks = [merge(chunk, pair, new_id) for chunk in ids_chunks]

            merges[pair] = new_id
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
            
            if verbose:
                print(f"{pair} -> {new_id}")

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode(encoding="utf-8", errors="replace"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode(encoding="utf-8", errors="replace")
        return text
    
    def _encode_chunk(self, text_bytes):
        """
        就是正常的encode，只不过这里没有对于special tokens的处理
        """
        ids = list(text_bytes)
        while len(ids) >= 2:
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids
    
    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids
    
    def encode(self, text, allowed_special="none_raise"):
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)

        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids
    

    
