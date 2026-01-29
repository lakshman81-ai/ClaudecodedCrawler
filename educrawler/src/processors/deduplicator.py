"""
Content deduplicator using MinHash for similarity detection.

Usage:
    from processors.deduplicator import ConceptDeduplicator

    deduplicator = ConceptDeduplicator()
    unique_concepts = deduplicator.deduplicate_concepts(concepts)
"""

from __future__ import annotations

import re
from typing import List
from datasketch import MinHash, MinHashLSH

from src.models.content import Concept


class ConceptDeduplicator:
    """
    Deduplicator for Concept objects using MinHash similarity detection.

    Attributes:
        threshold: Similarity threshold (0.0-1.0). Default 0.85 = duplicate
        num_perm: Number of permutations for MinHash. Higher = more accurate
    """

    def __init__(self, threshold: float = 0.85, num_perm: int = 128):
        """
        Initialize deduplicator.

        Args:
            threshold: Jaccard similarity threshold for duplicates (0.85 = 85% similar)
            num_perm: Number of permutations for MinHash accuracy
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self.threshold = threshold
        self.num_perm = num_perm

    def deduplicate_concepts(self, concepts: List[Concept]) -> List[Concept]:
        """
        Remove duplicate concepts based on content similarity.

        Uses MinHash LSH (Locality-Sensitive Hashing) to efficiently find
        near-duplicate concepts. Keeps the first occurrence of each unique concept.

        Args:
            concepts: List of Concept objects to deduplicate

        Returns:
            List of unique Concept objects
        """
        if not concepts:
            return []

        if len(concepts) == 1:
            return concepts

        # Create LSH index for efficient similarity search
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        unique_concepts = []
        seen_hashes = set()

        for concept in concepts:
            # Generate MinHash for this concept
            minhash = self._create_minhash(concept)

            # Check if similar concept already exists
            similar = lsh.query(minhash)

            if not similar:
                # This is a unique concept
                concept_id = str(concept.id)
                lsh.insert(concept_id, minhash)
                seen_hashes.add(concept_id)
                unique_concepts.append(concept)

        return unique_concepts

    def _create_minhash(self, concept: Concept) -> MinHash:
        """
        Create MinHash signature for a Concept.

        Combines title, definition, explanation, and key points into
        a set of tokens for similarity comparison.

        Args:
            concept: Concept to hash

        Returns:
            MinHash signature
        """
        minhash = MinHash(num_perm=self.num_perm)

        # Combine all text fields
        combined_text = " ".join([
            concept.title,
            concept.definition,
            concept.explanation,
            " ".join(concept.key_points)
        ])

        # Tokenize and add to MinHash
        tokens = self._tokenize(combined_text)
        for token in tokens:
            minhash.update(token.encode('utf-8'))

        return minhash

    def _tokenize(self, text: str) -> set[str]:
        """
        Tokenize text into set of normalized words.

        Args:
            text: Text to tokenize

        Returns:
            Set of normalized tokens
        """
        # Convert to lowercase
        text = text.lower()

        # Remove special characters except spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Split into words and remove short tokens
        tokens = {word for word in text.split() if len(word) > 2}

        return tokens

    def calculate_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """
        Calculate Jaccard similarity between two concepts.

        Args:
            concept1: First concept
            concept2: Second concept

        Returns:
            Similarity score between 0.0 and 1.0
        """
        minhash1 = self._create_minhash(concept1)
        minhash2 = self._create_minhash(concept2)

        return minhash1.jaccard(minhash2)


def deduplicate_concepts(concepts: List[Concept], threshold: float = 0.85) -> List[Concept]:
    """
    Convenience function to deduplicate concepts.

    Args:
        concepts: List of concepts to deduplicate
        threshold: Similarity threshold (default 0.85)

    Returns:
        List of unique concepts
    """
    deduplicator = ConceptDeduplicator(threshold=threshold)
    return deduplicator.deduplicate_concepts(concepts)
