from collections.abc import Sequence


def char_to_utf8_bytes(text:str):
    return text.encode("utf-8")
    
def get_par_counts(input: Sequence[int]):
    pairs = list(zip(input,input[1:]))
    counts = dict.fromkeys(pairs, 0)

    for item in pairs:
        #print(item)
        counts[item] = counts[item] + 1 

    return counts    

def merge(input: Sequence[int], pair: tuple[int, int], next_token: int):
    tokens = list(input)

    x = 0
    while x < len(tokens) - 1:
        if tokens[x] == pair[0] and tokens[x+1] == pair[1]:
            tokens[x:x + 2] = [next_token]
            #copy[x] = counter
            #del copy[x+1]
           
        x+=1

    return tokens

def train_bpe(text: str, iterations: int):
    tokens = list(char_to_utf8_bytes(text))
    merges = {}
    # for i in range(0,256):
    #     merges[i] = i

    x = 0
    next_token = 256
    while x < iterations:
        counts = get_par_counts(tokens)
        if len(counts) == 0:
            break

        sorted_counts = sorted(counts.items(), key = lambda item: item[1],reverse= True)
        tokens = merge(tokens, sorted_counts[0][0], next_token)
        merges[sorted_counts[0][0]] = next_token
        
        x+=1
        next_token+=1

    return tokens , merges

def encode(text: str, merges: dict[tuple[int,int], int]) -> list[int]:
    tokens = list(char_to_utf8_bytes(text))

    #merges is sorted already
    for key in merges:
        x = 0
        while x < len(tokens) - 1:
            if tokens[x] == key[0] and tokens[x+1] == key[1]:
                tokens[x:x + 2] = [merges[key]]            
            x+=1

    return tokens

def build_vocab( merges: dict[tuple[int,int], int]) -> dict[int, bytes]:
    vocab = dict[int, bytes]()
    for c in range(256):
        vocab[c] = bytes([c])

    #merges is sorted already
    for key in merges:
        vocab[merges[key]] = vocab[key[0]] + vocab[key[1]]

    return vocab

def decode(tokens: list[int], vocab:dict[int, bytes]) -> str:
    final_tokens = bytes()

    for token in tokens:
        final_tokens += vocab[token]

    return final_tokens.decode("utf-8")



def main():

    # utf = char_to_utf8_bytes("abab")
    # counts = get_par_counts(utf)
    
    # #print(counts.items())

    # merged = merge(utf, [97,98])
    # print(merged)

    tokens,  merges = train_bpe("abab bb ba", 1)
    # print(tokens)
    print(merges)

    encoded = encode("habe jłąd dd", merges)
    print(encoded)

    vocab = build_vocab(merges)
    print(vocab)

    d = decode(encoded, vocab)
    print(d)

if __name__ == "__main__":
    main()
