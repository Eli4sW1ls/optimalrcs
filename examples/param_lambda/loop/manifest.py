"""Pass manifests: config, seeds and hashes of the lambda model, the interfaces and the data (constraint 8)."""
import hashlib
import json
import os
import time

import numpy as np


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_of(obj):
    """md5 of a JSON-serializable object (interfaces, config)."""
    return hashlib.md5(json.dumps(jsonable(obj), sort_keys=True).encode()).hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def write(path, **fields):
    data = dict(fields)
    data.setdefault('written', time.strftime('%Y-%m-%d %H:%M:%S'))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(jsonable(data), f, indent=1)
    return data


def read(path):
    with open(path) as f:
        return json.load(f)


def update(path, **fields):
    data = read(path) if os.path.isfile(path) else {}
    data.update(fields)
    return write(path, **data)
