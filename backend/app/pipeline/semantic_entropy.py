"""Extra uncertainty signal (Farquhar et al., Nature 2024) - built and tested, wired in next phase.

Idea: ask the LLM the same question several times. Group the answers that MEAN the same thing
(each entails the other). If the answers scatter over many different meanings, the model is
guessing -> likely a confabulation.

It needs an LLM that can be sampled several times (API key or a local model), so the /analyze
route does not call it yet. The clustering + entropy maths below is complete and unit-tested.
"""
import math


def cluster_by_meaning(answers, entails):
    """Greedy bidirectional-entailment clustering. `entails(a, b)` -> bool."""
    clusters = []
    for answer in answers:
        for cluster in clusters:
            representative = cluster[0]
            if entails(representative, answer) and entails(answer, representative):
                cluster.append(answer)
                break
        else:
            clusters.append([answer])
    return clusters


def semantic_entropy(answers, entails):
    """Discrete semantic entropy (in nats). 0 = all answers agree; higher = more uncertain."""
    if len(answers) < 2:
        return 0.0
    clusters = cluster_by_meaning(answers, entails)
    n = len(answers)
    return -sum((len(c) / n) * math.log(len(c) / n) for c in clusters)
