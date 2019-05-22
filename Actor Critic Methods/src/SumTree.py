"""
*** Special thanks to Jaromir Janish for the Deep-Q-Network tutorial he put together. His code provided a fantastic
starting place for designing my reinforcment learning experiments.
Link: https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/ ***

"""

import numpy


class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2*capacity - 1)  # The size of the binary tree
        self.data = numpy.zeros(capacity, dtype=object)  # Size of data containing experiences

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2  # Tree is filled left to right

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        Search throught the nodes
        """
        left = 2 * idx + 1
        right = left + 1

        # If we reach the bottom of the tree
        if left >= len(self.tree):
            return idx

        # Downwards search for higher priority node
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        """
        Add priority score in the sumtree leaf and add the experience in data
        """
        idx = self.write + self.capacity - 1  # Look at what index we want to put the experience

        self.data[self.write] = data  # Update the dataframe
        self.update(idx, p)  # Update the leaf

        self.write += 1

        # If we're above the capacity, you go back to first index (we overwrite)
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        """
        Update the leaf priority score and propagate the change through tree.
        """
        change = p - self.tree[idx]  # Change = new priority score - former priority score

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """
        Get the leaf_index, priority value of that leaf and experience associated with that index.
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return idx, self.tree[idx], self.data[dataIdx]