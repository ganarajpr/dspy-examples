from similarity import sentence_similarity


def similarity(example, pred, trace=None):
    sim = sentence_similarity(example.english, pred.english)
    return sim > 0.7
