import tensorflow as tf
import os
import numpy as np


def create_np_memmap_file(path, column_size, row_size):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.memmap(path, dtype='float32', mode='w+', shape=(column_size, row_size))


def update_column_to_memmap_file(path_with_name, data, columnumber, column_size, row_size):
    data = data.ravel()
    mem = np.memmap(path_with_name, dtype='float32', mode='r+', shape=(column_size, row_size))
    mem[:, columnumber] = data
    mem.flush()
    # z.append(data, axis=1)
    print(columnumber)




