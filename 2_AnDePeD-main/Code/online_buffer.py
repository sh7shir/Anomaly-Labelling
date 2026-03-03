import numpy as np

import collections
import itertools


class CircularBuffer:
    def __init__(self, size: int):
        self.size = size
        self.buffer = collections.deque(maxlen=size)
        return

    def load(self, data: list):
        """
        Clear then load buffer with data.
        If data is longer then the size of the buffer, only the last N points will be in the buffer (if N=len(data)).
        :param data: list of values to load into buffer
        :return:
        """
        self.buffer.clear()
        self.buffer.extend(data)
        return

    def add_item(self, item):
        """
        Add the item into the buffer.
        If the buffer is already full, the first item is dropped to make way for the new one.
        :param item: item to add
        :return:
        """
        self.buffer.append(item)
        return

    def get_last_n_items(self, n: int):
        """
        Returns the last n items from the buffer.
        :param n: how many items
        :return:
        """
        begin = self.size - n
        end = self.size
        return list(itertools.islice(self.buffer, begin, end))

    def get_last_item(self):
        return self.buffer[-1]

    def get_all_items(self):
        to_ret = np.asarray(self.buffer)
        return to_ret

    def get_size(self):
        return self.size
