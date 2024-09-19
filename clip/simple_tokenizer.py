import gzip
import html
import os
from functools import lru_cache

import ftfy  # fixed that for you
import regex as re

@lru_cache()
def default_bpe():   # bpe text file을 가져오는 함수
    return os.path.join(os.path.dirnmae(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")

@lru_cache()
def bytes_to_unicode(): # gz파일을 decode 하는 함수
    """
    Return list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 100 token dataset you end up needing around 5K for decent coverage.
    This is a significant percent of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mappingto whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1)) + 
        list(range(ord("i"), ord("-")+1)) +
        list(range(ord("o"), ord("y")+1)) 
    # bs는 모든값
    cs = bs[:] # bs에 1대 1대응하는 unicode 값을 임의로 저장하는 변수, 256부터 시작해서 +1씩 증가
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):  # 단어를 pair별로 내뱉어줌
    """
    Return set of symbol pairs in a word. Word is represented as tuple of symbols (symbols being variable-length strings).
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescpae(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}  # 기존 33: '!' -> '!': 33 형태로 나옴, 난 이걸 byte_encoder2라는 변수에 지정
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')  # 출력하면 ['i n', ''] ['t h', ''] 이런 형태로 나옴
        merges = merges[1:49152-256-2+1] # 길이가 262146 -> 48894로 줄어듦, remove한 text들 save해놓긴 했음, 무슨 규칙으로 text들을 지웠는지 몰겠음!
        merges = [tuple(merge.split()) for merge in merges]  # 위에까지 ['i n']인 값이 split을 통해 ('i', 'n')으로 출력, 이걸 전부다시 dict로 출력
        vocab = list(bytes_to_unicode().values())  # byte_encoder2의 값인 '!': 33 에서 숫자값만 가져옴
        vocab = vocab + [v+'</w>' for v in vocab]  # 한줄로 출력해줌 
        for merge in merges:
            vocab.append(''.join(merge)) # append
        vocab.extend(['<|startoftext|>', '<|endoftext|>']) # 앞뒤로 extend
        self.encoder = dict(zip(vocab, range(len(vocab))))  # encoder_h라는 변수로 저장함, vocab과 길이를 mapping해서 전체 dict로 만듦
        self.decoder = {v:k for k, v in self.encoder.items()} # encoder_h를 반대로 뽑아냄
        self.bpe_ranks = dict(zip(merges, range(len(merges)))) # 결과값이 "{('i', 'n'): 0, ('t', 'h'): 1, ('a', 'n'): 2 ...} 등 rank를 매겨서 출력
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}  # dict형태로 출력
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)
        # 텍스트에서 단어, 숫자, 특수문자 등을 추출

    def bpe(self, token):
        # dict형태로 붙여진 token
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1])+(token[-1] + '</w>',) # 문장의 끝에 </w>를 이어붙이기 위해 [:-1]과 [-1]을 이어붙임, 튜플형식으로
        pairs = get_pairs(word)  # token을 이어붙인 word를 [0] [1:] pair로 묶음

        """ 주어진 토큰을 BPE(Byte Pair Encoding) 방식으로 인코딩하는 함수, 
        BPE는 하나의 문장이 모두 이어져 있는 것이 아닌 띄어쓰기 등으로 분리되어 있다고 가정             
        """

        if not pairs:
            return token+'</w>' 
        
        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))  # pair가 self.bpe_ranks안에 존재하는지 확인하고, 있다면 반환, 없다면 기본값 float('inf') 무한대를 반환
            # min()을통해 iterable에서 가장 작은 값을 반환
            # bigram의 출력물은 ('c', 'h') <-- 이런 형태로 반환됨
            if bigram not in self.bpe_ranks:  
                break
            first, second = bigram  # 출력물 예시: (c , h) 에서 c가 first, h가 second
            new_word = [] # 병합하는 과정, 단어 형태로 출력됨
            i = 0
            while i < len(word): 
                try:
                    j = word.index(first, i)  # first의 index를 정해줌
                    # 첫 번째 문자가 나오기 전까지의 문자를 새로운 단어 리스트에 추가
                    new_word.extend(word[i:j]) 
                    i = j
                except: 
                    # 더 이상 찾을 수 없으면 나머지 문자를 모두 추가하고 종료
                    new_word.extend(word[i:])
                    break

                # 첫 번째 문자가 현재 위치에 있고, 다음 문자가 병합할 두 번째 문자라면 병합
                if word[i] == first and i <len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2   # 병합된 두 문자를 건너뜀
                else:
                    new_word.append(word[i])
                    i += 1
            
            # 새로운 단어 리스트를 튜플로 변환 (튜플을 사용하는 이유는 불변성 때문)
            new_word = tuple(new_word)
            word = new_word   # new_word를 word로 넣어줌
            if len(word) == 1:
                break
            else:
                # 병합할 새로운 문자 쌍을 다시 구함
                pairs = get_pairs(word)# 위에서 봤던 basic_clean을 거치고 소문자로 바꿔준 후 whitespace_clean을 통해 텍스트 정리
        
        # 최종 병합된 단어를 공백으로 구분된 문자열로 변환
        word = ' '.join(word)  # 다시 단어로 돌아옴, 'c h a i r'가 됨
        self.cache[token] = word  # 이걸 토큰처리하고 계속 list에 넣음
        return word  # word인 c h a i r를 return
    

    def encode(self, text):
        """
        BPE 토큰 -> 텍스트로 인코딩하는 함수
        텍스트를 clean 하고 각 단어를 BPE 방식으로 인코딩해서 BPE 토큰으로 변환
        """
        bpe_tokens = []        
        text = whitespace_clean(basic_clean(text)).lower() # 위에서 봤던 basic_clean을 거치고 소문자로 바꿔준 후 whitespace_clean을 통해 텍스트 정리
        for token in re.findall(self.pat, text): 
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))  # 각 토큰을 바이트 단위로 인코딩하고, 이를 유니코드 값으로 변환 (바이트 -> 유니코드 매핑 사용)
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))  # 매핑된 토큰에 ' ' 공간을 주고 리스트에 extend로 넣는 형식
        return bpe_tokens  # 최종반환될 token들은 byte단위로 이어붙인 ' '공간이 있는 character들의 나열, 이 character들이 모여서 text가 되겠지
    

    def decode(self, tokens):
        """
        BPE 토큰 -> 텍스트로 디코딩하는 함수
        BPE 방식으로 인코딩된 토큰을 원래의 텍스트로 복원
        """
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text
    

    # 여기서 생기는 근본적인 궁금점:
    # 1 어짜피 텍스트를 토큰으로 바꿔야 하는데, 그 토큰화 하는 방법을 BPE로 했고, 왜 굳이 다시 텍스트로 복원하지? 출력때문에 그런건가? ㅇㅇㅇㅇ맞음
    # 2 decode는 꼼꼼히 못봐서 상현이꺼 주석 복붙했음, 대략적으로 decode의 역할은 이해가 되지만 linebyline까지로는 숙지를 안했음, 잊지말것