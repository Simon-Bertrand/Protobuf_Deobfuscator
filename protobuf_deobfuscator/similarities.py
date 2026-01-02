

from typing import Counter, NamedTuple, Optional,List


def weighted_sum(
    weights: List[float],
    scores: List[float],
) -> float:
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)


class ExactSimilarityDefinition:
    @staticmethod
    def merge_hungarian_path_score(score, list1, list2) -> float:
        return score * ExactSimilarityDefinition.length_similarity(
            len(list1), len(list2)
        )

    @staticmethod
    def length_similarity(len1: int, len2: int) -> float:
        if len1 == len2:
            return 1.0
        return min(len1, len2) / max(len1, len2)

    @staticmethod
    def hard_length_similarity(len1: int, len2: int) -> float:
        max_len = min(len1, len2)
        if max_len == 0:
            return 0.0
        return max(0, 1.0 - (abs(len1 - len2) / max_len))

    @staticmethod
    def counter_similarity(types1: list, types2: list) -> float:
        if (
            out := ExactSimilarityDefinition.manage_empty_lists(types1, types2)
        ) is not None:
            return out
        c1 = Counter(types1)
        c2 = Counter(types2)

        # CHANGE 1: Calculate the size difference penalty (structural cost).
        # This factor is 1.0 if sizes are equal (modification) and < 1.0 if sizes differ (deletion).
        max_len = min(len(types1), len(types2))
        if max_len == 0:
            return 0.0
        structural_penalty = ExactSimilarityDefinition.hard_length_similarity(
            len(types1), len(types2)
        )

        # Retain original Intersection and Union calculation
        union_sum = sum((c1 | c2).values())
        if union_sum == 0:
            return 0.0

        jaccard_score = sum((c1 & c2).values()) / union_sum

        # CHANGE 2: Return the original Jaccard score multiplied by the Structural Penalty.
        # This ensures deletion (which changes the size) is always penalized more than modification (which keeps the size equal).
        return jaccard_score * structural_penalty ** (1 / 2)

    @staticmethod
    def manage_empty_lists(list1, list2) -> Optional[float]:
        if len(list1) == 0 or len(list2) == 0:
            return 0
        return None

    @staticmethod
    def manage_empty_weights(list1, list2) -> Optional[float]:
        return not (len(list1) == 0 and len(list2) == 0)



class SimilarityOutput(NamedTuple):
    score: float
    weight: float

    @staticmethod
    def hard_penalization() -> "SimilarityOutput":
        return SimilarityOutput(weight=1, score=-1)

    @staticmethod
    def null() -> "SimilarityOutput":
        return SimilarityOutput(weight=0, score=0.0)

    @staticmethod
    def zero() -> "SimilarityOutput":
        return SimilarityOutput(weight=1, score=0.0)

    @staticmethod
    def perfect() -> "SimilarityOutput":
        return SimilarityOutput(weight=1, score=1.0)