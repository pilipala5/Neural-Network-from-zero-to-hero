import unicodedata


def get_stats(ids: list, count=None) -> dict:
    """
    找到字节对的统计次数
    Example: [1, 2, 1, 2, 3] -> {(1, 2): 2, (2, 1): 1, (2, 3): 1}

    :params ids: list, 字节流
    :return count: dict, 字节对的统计次数
    """

    count = {} if count is None else count
    for (p0, p1) in zip(ids, ids[1:]):
        count[(p0, p1)] = count.get((p0, p1), 0) + 1
    
    return count


def merge(ids: list, pair: tuple, idx: int) -> list:
    """
    将字节对用最新idx替换
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx = 4 -> [4, 3, 4]

    :params ids: list, 原字节串
    :params pair: tuple, 原字节对
    :params idx: int, 新索引
    :return new_ids: list, 新字节串
    """

    new_ids = []

    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i = i + 2
        else:
            new_ids.append(ids[i])
            i = i + 1
    
    return new_ids


def replace_control_charactors(s: str) -> str:
    """
    去除字符串中的控制字符, 如"\n" 用unicode码表示
    Example: "hello \n world" -> "hello \u000a world"

    :params s: str, 原字符串
    :return : str, 新字符串
    """

    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] == "C":
            chars.append(f"\\u{ord(ch):04x}")
        else:
            chars.append(ch)
    
    return "".join(chars)


def render_token(t: bytes) -> str:
    """
    将字节流转换成str 并去除控制字符
    Example: 0x68 0x65 0x6c 0x6c 0x6f 0x20 0x0a 0x20 0x77 0x6f 0x72 0x6c 0x64 -> hello \u000a world

    :params t: bytes 字节流
    :return s: str 字符串
    """

    s = t.decode(encoding="utf-8", errors="replace")
    s = replace_control_charactors(s)
    
    return s


class Tokenizer():
    """Base class for Tokenizers"""


    def __init__(self):
        """
        Attributes:
            merges (dict): 存储合并的对和新ID的映射。
            vocab (dict): 存储字典，包含字符及其对应的字节表示。
            special_tokens(dict): 特殊字符
            pattern(str): 模式
        """
        self.merges = {}  # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {{'<|endoftext|>': 100257}}
        self.vocab = self._build_vocab() # int -> bytes
    
    def train(self, text, vocab_size, verbose=False):
        raise NotImplementedError
    
    def encode(self, text):
        raise NotImplementedError
        
    def decode(self, ids):
        raise NotImplementedError

    def _build_vocab(self):
        vocab = {idx: bytes(idx) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8", errors="replace")
        return vocab
    
    def save(self, file_prefix):
        # 保存模型文件，用于导入 
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            f.write(f"{len(self.special_tokens.items())}\n")
            
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        
        # 保存vocab 用于人工检查
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, 'w', encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                s = render_token(token)

                if idx in inverted_merges:
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        assert model_file.endswith(".model")

        # 读取model文件
        merges = {}
        special_tokens = {}
        idx = 256

        with open(model_file, 'r') as f:
            version = f.readline().strip()
            assert version == "minbpe v1"
            
            self.pattern = f.readline().strip()
            
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()