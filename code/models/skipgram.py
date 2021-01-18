"""Skipgram vector model, where we take the average/sum or other of vectors to
compute a sentence encoding."""
import numpy as np
from tqdm import tqdm


def load_vectors(vector_fn):
    print("Opening vector file...")
    with open(vector_fn, 'r') as vector_file:
        total_num = vector_file.readline().split()[0]
        print(f"Loading {total_num} vectors...")
        vector_dict = {ln.strip().split()[0]:
                       np.array(list(map(float, ln.strip().split()[1:])))
                       for ln in tqdm(vector_file.readlines())}
    return total_num, vector_dict


class AverageSkipgramModel():
    def __init__(self, fn_path):
        self.fn_path = fn_path
        self.size, self.vectors = load_vectors(self.fn_path)

    def model_sentence(self, sentence: str) -> np.ndarray:
        vecs = [self.vectors[w] for w in sentence.split() if w in self.vectors]
        return np.average(vecs, axis=0)
