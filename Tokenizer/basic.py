from base import Tokenizer, get_stats, merge


class BasicTokenizer(Tokenizer):
    """
    最简单的BPE进行分词

    
    """

    def __init__(self):
        super().__init__()
    
    def train(self, text, vocab_size, verbose=False):
        """
        对text进行训练，通过BPE得到merge

        params text: str, 文本训练内容
        params vocab_size: int(>=256), 得到merge个数为(vocab_size - 256)
        params verbose: bool, 是否打印
        """

        assert vocab_size >= 256
        num_merges = vocab_size - 256
        idx = 256
        
        text_bytes = text.encode("utf-8", "replace")
        ids = list(text_bytes)

        vocab = {i : bytes([i]) for i in range(256)}
        merges = {}
        for i in range(num_merges):
            new_id = idx + i
            stats = get_stats(ids)

            pair = max(stats, key=stats.get)

            merges[(pair)] = new_id
            ids = merge(ids, pair, new_id)
            vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]

            if verbose:
                print(f"merge {i + 1}/{num_merges}:{pair} -> {idx}erges: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        self.merges = merges
        self.vocab = vocab

    def encode(self, s):
        s_bytes = s.encode(encoding="utf-8", errors="replace")
        ids = list(s_bytes)

        while len(ids) >= 2:
            stats = get_stats(ids)
            # merge需要按照训练时的先后顺序

            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break

            ids = merge(ids, pair, self.merges[pair])

        return ids
    
    def decode(self, ids):
        t = b"".join(self.vocab[i] for i in ids)
        return t.decode(encoding="utf-8", errors="replace")
    

if __name__ == "__main__":
    tokenizer = BasicTokenizer()
    text = "aaabdaaabac"
    tokenizer.train(text, 256+3)
    print(tokenizer.encode(text))
    print(tokenizer.decode([258, 100, 258, 97, 99]))
    tokenizer.save("toy")
