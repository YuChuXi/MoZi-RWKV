import os
import pathlib
from typing import List, Set, Tuple, Callable

# Taken from https://github.com/BlinkDL/ChatRWKV/tree/main/tokenizer/rwkv_tokenizer.py

class Trie:
    __slots__ = ('ch', 'to', 'values', 'front')

    def __init__(self, front=None, ch=None) -> None:
        self.ch = ch
        self.to: List = [None for _ in range(256)]
        self.values: Set = set()
        self.front = front

    def add(self, key: bytes, idx: int = 0, val=None) -> 'Trie':
        if idx == len(key):
            if val is None:
                val = key

            self.values.add(val)

            return self

        ch = key[idx]

        if self.to[ch] is None:
            self.to[ch] = Trie(front=self, ch=ch)

        return self.to[ch].add(key, idx=idx + 1, val=val)

    def find_longest(self, key: bytes, idx: int = 0) -> Tuple[int, 'Trie', set]:
        u: Trie = self
        ch: int = key[idx]
        ret = None

        while u.to[ch] is not None:
            u = u.to[ch]
            idx += 1

            if u.values:
                ret = idx, u, u.values

            if idx == len(key):
                break

            ch = key[idx]

        assert ret is not None, 'Entry not found'

        return ret

    def __repr__(self) -> str:
        fr = self
        ret = []

        while fr is not None:
            if fr.ch is not None:
                ret.append(fr.ch)

            fr = fr.front

        return '<TRIE %s %s>' % (ret[::-1], self.values)

class WorldTokenizer:

    def __init__(self, file_path) -> None:
        self.index_to_token = {}

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        for line in lines:
            idx = int(line[:line.index(' ')])
            x = eval(line[line.index(' '):line.rindex(' ')])
            x = x.encode('utf-8') if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(line[line.rindex(' '):])
            self.index_to_token[idx] = x

        self.token_to_index = {}

        for k, v in self.index_to_token.items():
            self.token_to_index[v] = int(k)

        self.root = Trie()

        for t, i in self.token_to_index.items():
            _ = self.root.add(t, val=(t, i))

    def encode_bytes(self, src: bytes) -> List[int]:
        idx: int = 0
        tokens: List[int] = []

        while idx < len(src):
            _idx: int = idx
            idx, _, values = self.root.find_longest(src, idx)
            assert (idx != _idx)
            _, token = next(iter(values))
            tokens.append(token)

        return tokens

    def decode_bytes(self, tokens: List[int]) -> bytes:
        return b''.join(map(lambda i: self.index_to_token[i], tokens))

    def encode(self, src: str) -> List[int]:
        return self.encode_bytes(src.encode('utf-8'))

    def decode(self, tokens: List[int]) -> str:
        # 'replace' error handling mode will insert \uFFFD characters in place of malformed/partial UTF-8 sequences.
        # Downstream code needs to detect \uFFFD and attempt to postpone decoding until more tokens arrive and UTF-8 sequences are complete.
        return self.decode_bytes(tokens).decode('utf-8', errors='replace')

def get_world_tokenizer_v20230424() -> Tuple[
    Callable[[List[int]], str],
    Callable[[str], List[int]]
]:
    """
    Loads the default World tokenizer, commonly used in RWKV v4 World models.
    Returns a tuple of `decode` and `encode` functions.
    """
    parent: pathlib.Path = pathlib.Path(os.path.abspath(__file__)).parent
    tokenizer: WorldTokenizer = WorldTokenizer(parent / 'rwkv_vocab_v20230424.txt')
    return tokenizer.decode, tokenizer.encode


class RWKV_TOKENIZER():
    table: List[List[List[bytes]]]
    good: List[Set[int]]
    wlen: List[int]
    def __init__(self, file_name):
        self.idx2token = {}
        sorted = [] # must be already sorted
        lines = open(file_name, "r", encoding="utf-8").readlines()
        for l in lines:
            idx = int(l[:l.index(' ')])
            x = eval(l[l.index(' '):l.rindex(' ')])
            x = x.encode("utf-8") if isinstance(x, str) else x
            assert isinstance(x, bytes)
            assert len(x) == int(l[l.rindex(' '):])
            sorted += [x]
            self.idx2token[idx] = x

        self.token2idx = {}
        for k, v in self.idx2token.items():
            self.token2idx[v] = int(k)

        # precompute some tables for fast matching
        self.table = [[[] for j in range(256)] for i in range(256)]
        self.good = [set() for i in range(256)]
        self.wlen = [0 for i in range(256)]

        for i in reversed(range(len(sorted))): # reverse order - match longer tokens first
            s = sorted[i]
            if len(s) >= 2:
                s0 = int(s[0])
                s1 = int(s[1])
                self.table[s0][s1] += [s]
                self.wlen[s0] = max(self.wlen[s0], len(s))
                self.good[s0].add(s1)

    def encodeBytes(self, src: bytes) -> List[int]:
        src_len: int = len(src)
        tokens: List[int] = []
        i: int = 0
        while i < src_len:
            s: bytes = src[i : i + 1]

            if i < src_len - 1:
                s1: int = int(src[i + 1])
                s0: int = int(src[i])
                if s1 in self.good[s0]:
                    sss: bytes = src[i : i + self.wlen[s0]]
                    try:
                        s = next(filter(sss.startswith, self.table[s0][s1]))
                    except:
                        pass
            tokens.append(self.token2idx[s])
            i += len(s)

        return tokens

    def decodeBytes(self, tokens):
        return b''.join(map(lambda i: self.idx2token[i], tokens))

    def encode(self, src: str):
        return self.encodeBytes(src.encode("utf-8"))

    def decode(self, tokens):
        return self.decodeBytes(tokens).decode('utf-8')

    def printTokens(self, tokens):
        for i in tokens:
            s = self.idx2token[i]
            try:
                s = s.decode('utf-8')
            except:
                pass
            print(f'{repr(s)}{i}', end=' ')
            # print(repr(s), i)
        print()
