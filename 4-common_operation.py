import numpy as np

def dict_sort(content, K=None, reverse=True):
    results = {}
    sort_info = sorted(content.items(), key=lambda item: item[1], reverse=reverse)
    for idx, info in enumerate(sort_info):
        if K is None:
            results[info[0]] = info[1]
        else:
            if idx < K:
                results[info[0]] = info[1]
            else:
                break

    return results


# tanh function by numpy
def tanh(x):
    s1 = np.exp(x) - np.exp(-x)
    s2 = np.exp(x) + np.exp(-x)
    s = s1 / s2
    return s


# L2 loss function by numpy
def L2(yhat, y):
    loss = np.sum(np.multiply((y - yhat), (y - yhat)))
    return loss