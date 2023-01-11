from collections import Mapping


def iter_leaves(obj):
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            for k, v in iter_leaves(value):
                k.insert(0, key)
                yield (k, v)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            for k, v in iter_leaves(value):
                k.insert(0, i)
                yield (k, v)
    else:
        yield [], obj


def set_leaf(obj, keys, value):
    for k in keys[:-1]:
        obj = obj[k]
    obj[keys[-1]] = value
