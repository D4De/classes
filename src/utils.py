
from typing import Dict, List, Tuple, TypeVar
import numpy as np

K = TypeVar("K")
V = TypeVar("V")

def unpack_table(table : Dict[K,V]) -> Tuple[List[K], List[V]]:
    """
    Given a lookup table, implemented as a dictionary, it separates the keys from values
    and returns them in pairs but in different lists.
    Arguments:
        table {dict} -- Lookup table.

    Returns:
        [list, list] -- Returns two lists, the first one contains the keys while the
        latter contains the values.
    """
    keys = []
    values = []
    # Move each pair of key and value to two separate
    # lists to keep the order.
    for key, value in table.items():
        keys.append(key)
        values.append(value)
    return keys, values


def random_choice(a, size = None, replace = None, p = None):
    """
    Wrapper for numpy choice function that accepts probabilities that do not sum to
    one. p will be normalized.
    This function will also mitigate the risk of raising ValueError numerical errors.
    See np.random.choice for better explanation of the parameters
    """
    if p is not None:
        p1 = np.asarray(p).astype('float64')
        p1 = p1 / np.sum(p1)
        p = p1
    return np.random.choice(a, size=size, replace=replace, p=p)