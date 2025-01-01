import abc
import random
from itertools import permutations
from typing import Set


import itertools
from random import shuffle, randint
from typing import List, Tuple


class KSumDataset(abc.ABC):
    def __init__(self, p: int, k: int, frac_train: float,seed: int = None):
        """
        Initialize the KSumDataset with multiple groups of elements and k-tuple combinations.
        , group_elements: List[Set[int]]
        :param p: The modulo value for sum operation.
        :param group_elements: A list of sets, each set represents a group of unique elements.
        :param k: The number of elements to combine from the groups.
        :param frac_train: Fraction of data to be used
         for training (rest for validation).
        """
        if seed is not None:
            random.seed(seed)
        self.p = p
        # self.group_elements = group_elements
        self.k = k
        self.frac_train = frac_train
        self.idx2vocab = ['o', '='] + list(range(p))
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = p+2
        self.n_out = p
        idxs = self._generate_combinations()
        random.shuffle(idxs)
        self.train_pairs, self.val_pairs = idxs[:int(len(idxs) * frac_train)], idxs[int(len(idxs) * frac_train):]

    def _generate_combinations(self) -> List[Tuple[int]]:
        """
        Generate all possible combinations of k elements across the provided groups.
        """
        # Assuming each element from each group can be chosen independently.
        # This might not be the intended behavior if combinations must respect group boundaries.
        all_combinations = list(itertools.product(list(range(self.p)),repeat=self.k))
        return all_combinations

    def encode(self, sequence):
        return [self.vocab2idx[item] for item in sequence]

    def decode(self, sequence):
        return [self.idx2vocab[item] for item in sequence]

    def fetch_output(self, elements_list: List[int]) -> int:
        return sum(elements_list) % self.p

    def form_equation(self, elements: List[int], c):
        equation_list = [elements[0]]
        for i in range(1,len(elements)):
            equation_list += ['o',elements[i]]
        equation_list += ['=',c]
        return equation_list

    def fetch_example(self, idx):
        """
        Fetch a single example by index for training or validation.
        """
        elements = list(idx)
        c = self.fetch_output(elements)
        equation = self.form_equation(elements,c)
        return self.encode(equation[:-1]), (self.vocab2idx[c]-2), equation


    def fetch_train_example(self):
        idx = random.choice(self.train_pairs)
        return self.fetch_example(idx)


    def fetch_val_example(self):
        idx = random.choice(self.val_pairs)
        return self.fetch_example(idx)



# dataset = KSumDataset(p=3, k=3,frac_train=0.4)
# train_example = dataset.fetch_train_example()
# val_example = dataset.fetch_val_example()
# print("Training Example:", train_example)
# print("Validation Example:", val_example)
# print('start:',dataset.train_pairs)
# print(dataset.val_pairs)


class AbstractDataset(abc.ABC):
    """

    """
    def __init__(self, group_elements1: Set, group_elements2: Set, frac_train: float,seed: int = None):

        self.frac_train = frac_train
        self.group_elements1 = group_elements1
        self.group_elements2 = group_elements2
        self.ordered_group_elements1 = list(self.group_elements1)
        self.ordered_group_elements2 = list(self.group_elements2)
        self.idx2vocab = ['o', '='] + list(group_elements1.union(group_elements2))
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)
        self.n_out = len(group_elements1.union(group_elements2))
        idxs = list(range(len(self.group_elements1)*len(self.group_elements2)))
        random.shuffle(idxs)
        self.train_pairs, self.val_pairs = idxs[:int(len(idxs)*frac_train)], idxs[int(len(idxs)*frac_train):]
    
    @abc.abstractmethod
    def fetch_output(self, a, b):
        pass

    def encode(self, sequence):
        return [self.vocab2idx[item] for item in sequence]
    
    def decode(self, sequence):
        return [self.idx2vocab[item] for item in sequence]
    
    def form_equation(self, a, b, c):
        return [a, 'o', b, '=', c]
    
    def fetch_example(self, idx):
        a = self.ordered_group_elements1[idx // len(self.group_elements2)]
        b = self.ordered_group_elements2[idx % len(self.group_elements2)]
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation[:-1]), (self.vocab2idx[c]-2), equation
    
    def fetch_train_example(self):
        idx = random.choice(self.train_pairs)
        return self.fetch_example(idx)

    def fetch_val_example(self):
        idx = random.choice(self.val_pairs)
        return self.fetch_example(idx)

class ModSumDataset(AbstractDataset):
    """
    Summary of ModSumDataset
    -----------------------


    Detailed description of ModSum"""
    def __init__(self, p, frac_train,seed: int = None):
        super(ModSumDataset, self).__init__(set(range(p)), set(range(p)), frac_train,seed)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a + b) % self.p

class ModSubtractDataset(AbstractDataset):
    """

    """
    def __init__(self, p, frac_train):
        super(ModSubtractDataset, self).__init__(set(range(p)), set(range(p)), frac_train)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a * b) % self.p

class ModDivisonDataset(AbstractDataset):
    """

    """
    def __init__(self, p, frac_train):
        super(ModDivisonDataset, self).__init__(set(range(p)), set(range(1, p)), frac_train)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a * pow(b, self.p-2, self.p)) % self.p

class PermutationGroup(AbstractDataset):
    """

    """
    def __init__(self, k, frac_train):
        perms = set(map(tuple, permutations(list(range(k)))))
        super(PermutationGroup, self).__init__(perms, perms, frac_train)
        self.k = k

    def fetch_output(self, a, b):
        return tuple([a[b[i]] for i in range(len(b))])