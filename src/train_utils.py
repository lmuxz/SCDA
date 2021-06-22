import time
import numpy as np

def extend_dataset(xs, xt):
    ns = xs.shape[0]
    nt = xt.shape[0]
    n = max(ns, nt)
    indexs = np.random.choice(range(ns), n, replace=True)
    indext = np.random.choice(range(nt), n, replace=True)
    return_s = list(range(ns)) + list(indexs[ns:n])
    return_t = list(range(nt)) + list(indext[nt:n])
    return return_s, return_t


def reduce_dataset(xs, xt):
    ns = xs.shape[0]
    nt = xt.shape[0]

    if ns > nt:
        indexs = np.random.choice(ns, nt, replace=False)
        indext = np.arange(nt)
    else:
        indexs = np.arange(ns)
        indext = np.random.choice(nt, ns, replace=False)
    return indexs, indext


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000), flush=True)
        return result
    return timed


def sample_validation_data(task, label, label_ratio=0.05, number_examples=None, *args, **kwargs):
    if number_examples is not None:
        sample_index = np.random.choice(label.shape[0], int(number_examples), replace=False)
    else:
        sample_index = np.random.choice(label.shape[0], int(label.shape[0] * label_ratio), replace=False)
    label = label[sample_index, 1]
    return sample_index, label


def balance_sample(label, n_examples, ratio_pos=None, *args, **kwargs):
    if ratio_pos is None:
        ratio_pos = label[:,1].mean()

    n_pos = int(ratio_pos * n_examples)
    n_neg = n_examples - n_pos

    index_pos = np.random.choice(np.where(label[:, 1]==1)[0], n_pos, replace=False)
    index_neg = np.random.choice(np.where(label[:, 1]==0)[0], n_neg, replace=False)

    index = np.r_[index_pos, index_neg]
    np.random.shuffle(index)
    return index
