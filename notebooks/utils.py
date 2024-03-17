"""
Bag-Of-Concepts utilities implementation code from the authors of the original paper
Github Link: https://github.com/hank110/bagofconcepts/blob/master/bagofconcepts/utils.py
"""

import numpy as np
from gensim.models import KeyedVectors, Word2Vec


def load_gensim_w2v(model_path, binary=False):
    return KeyedVectors.load_word2vec_format(model_path, binary=binary)


def train_gensim_w2v(
    corpus,
    embedding_dim,
    context,
    min_freq,
    iterations,
    save_path="",
    return_model=False,
):
    model = Word2Vec(
        vector_size=embedding_dim, window=context, min_count=min_freq, sg=1
    )
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=iterations)

    if save_path:
        model_name = "/w2v_model_d%d_w%d" % (embedding_dim, context)
        model.wv.save_word2vec_format(save_path + model_name)

        model.save(save_path + f"/Word2Vec_d{embedding_dim}_w{context}")

    if return_model:
        return model

    return model.wv.vectors, model.wv.index_to_key


class CustomWord2VecLoader:
    @classmethod
    def load_word2vec_format(cls, fname, fvocab=None, binary=False, norm_only=True):
        counts = None
        if fvocab is not None:
            counts = {}
            with open(fvocab, "r", encoding="utf-8") as fin:
                for line in fin:
                    word, count = line.strip().split()
                    counts[word] = int(count)

        with open(fname, "rb") as fin:
            header = fin.readline().decode("utf-8")
            vocab_size, layer1_size = map(int, header.split())
            result = KeyedVectors(vector_size=layer1_size)

            if binary:
                binary_len = np.dtype(np.float32).itemsize * layer1_size
                result.vectors = []  # Initialize as an empty list
                for line_no in range(vocab_size):
                    word = []
                    while True:
                        ch = fin.read(1)
                        if ch == b" ":
                            break
                        if ch != b"\n":
                            word.append(ch)
                    word = b"".join(word).decode("latin-1")

                    if counts is None:
                        result.key_to_index[word] = line_no  # type: ignore
                    elif word in counts:
                        result.key_to_index[word] = line_no  # type: ignore
                    else:
                        result.key_to_index[word] = line_no  # type: ignore

                    result.index_to_key.append(word)  # type: ignore
                    result.vectors.append(
                        np.frombuffer(fin.read(binary_len), dtype=np.float32)
                    )
                result.vectors = np.array(result.vectors)  # Convert to NumPy array
            else:
                for line_no, line in enumerate(fin):
                    parts = line.strip().split()
                    if len(parts) != layer1_size + 1:
                        raise ValueError(
                            "Invalid vector on line %s (is this really the text format?)"
                            % (line_no)
                        )
                    word, weights = parts[0], list(map(float, parts[1:]))

                    if counts is None:
                        result.key_to_index[word] = line_no  # type: ignore
                    elif word in counts:
                        result.key_to_index[word] = line_no  # type: ignore
                    else:
                        result.key_to_index[word] = line_no  # type: ignore

                    result.index_to_key.append(word)  # type: ignore
                    result.vectors.append(np.array(weights, dtype=np.float32))  # type: ignore
                result.vectors = np.array(result.vectors)  # Convert to NumPy array

        result.init_sims(norm_only)
        return result
