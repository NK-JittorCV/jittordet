from collections import Mapping


def iter_leaves(obj):
    """A generator to visit all leaves of obj."""
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
    if isinstance(keys, str):
        keys = keys.split('.')
    for k in keys[:-1]:
        obj = obj[k]
    obj[keys[-1]] = value


def delete_node(obj, keys):
    if isinstance(keys, (tuple, list)):
        for key in keys[:-1]:
            obj = obj[key]
        del obj[keys[-1]]
    else:
        assert isinstance(keys, str), 'only support search str node'
        if isinstance(obj, Mapping):
            if keys in obj:
                del obj[keys]
            for value in obj.values():
                delete_node(value, keys)
        elif isinstance(obj, (tuple, list)):
            for value in obj:
                delete_node(value, keys)
