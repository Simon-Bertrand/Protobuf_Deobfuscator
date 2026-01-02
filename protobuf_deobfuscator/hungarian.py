import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import NamedTuple


class HungarianPathGenerator:
    class Output(NamedTuple):
        P: np.ndarray
        S: np.ndarray
        row_ind: np.ndarray
        col_ind: np.ndarray
        score: float
        weight: float

    def __init__(self, method, normalized=True):
        self.normalized = normalized
        self.method = method

    def compute(self, list1, list2):
        if len(list1) == 0 or len(list2) == 0:
            return HungarianPathGenerator.Output(
                P=None, S=None, row_ind=None, col_ind=None, score=0, weight=0
            )
        sim_mat = np.zeros((len(list1), len(list2)))
        for i, el1 in enumerate(list1):
            for j, el2 in enumerate(list2):
                sim_mat[i, j] = self.method(el1, el2)
        row_ind, col_ind = linear_sum_assignment(-sim_mat)
        score = (
            (sim_mat[row_ind, col_ind].sum() / max(len(list1), len(list2)))
            if self.normalized
            else sim_mat[row_ind, col_ind].sum()
        )
        P = np.zeros_like(sim_mat)
        P[row_ind, col_ind] = 1
        return HungarianPathGenerator.Output(
            P=P, S=sim_mat, row_ind=row_ind, col_ind=col_ind, score=score, weight=1
        )