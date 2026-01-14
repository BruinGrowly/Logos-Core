"""
Hard Negatives - RLHF Learning Store
=====================================

Stores negative and positive associations learned from user corrections.
When a user corrects a system classification, we store:
- Hard Negative: (file_vector, wrong_concept) = reduce confidence
- Positive: (file_vector, correct_concept) = increase confidence

Storage: workspace/hard_negatives.json
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple


class HardNegatives:
    """
    Stores learned associations from user corrections.
    
    Uses vector similarity to apply confidence adjustments
    to future classifications.
    """
    
    # Confidence adjustment values
    NEGATIVE_ADJUSTMENT = -0.2  # Reduce confidence for known-bad associations
    POSITIVE_ADJUSTMENT = 0.1   # Boost confidence for known-good associations
    SIMILARITY_THRESHOLD = 0.85  # How similar a vector must be to trigger adjustment
    
    def __init__(self, store_path: str = None):
        """
        Initialize the hard negatives store.
        
        Args:
            store_path: Path to store file. Defaults to workspace/hard_negatives.json
        """
        if store_path is None:
            project_root = Path(__file__).parent.parent
            store_dir = project_root / "workspace"
            store_dir.mkdir(exist_ok=True)
            self.store_path = store_dir / "hard_negatives.json"
        else:
            self.store_path = Path(store_path)
        
        self._data = self._load_store()
    
    def _load_store(self) -> Dict:
        """Load store from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {'negatives': [], 'positives': []}
        return {'negatives': [], 'positives': []}
    
    def _save_store(self) -> None:
        """Save store to disk."""
        try:
            with open(self.store_path, 'w') as f:
                json.dump(self._data, f, indent=2)
        except IOError as e:
            print(f"[HardNegatives] Warning: Could not save store: {e}")
    
    def _vector_to_list(self, vector) -> List[float]:
        """Convert numpy array to list for JSON storage."""
        if hasattr(vector, 'tolist'):
            return vector.tolist()
        return list(vector)
    
    def _calculate_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            a = np.array(vec_a)
            b = np.array(vec_b)
            
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            return 0.0
    
    def add_negative(self, file_vector, wrong_concept: str, 
                     file_name: str = None) -> None:
        """
        Store a hard negative association.
        
        Args:
            file_vector: Vector embedding of the file content
            wrong_concept: The concept that was incorrectly matched
            file_name: Optional filename for logging
        """
        entry = {
            'vector': self._vector_to_list(file_vector),
            'concept': wrong_concept,
            'file_name': file_name,
            'type': 'negative'
        }
        
        self._data['negatives'].append(entry)
        self._save_store()
        
        print(f"[HardNegatives] Stored NEGATIVE: '{file_name}' should NOT match '{wrong_concept}'")
    
    def add_positive(self, file_vector, correct_concept: str,
                     file_name: str = None) -> None:
        """
        Store a positive reinforcement.
        
        Args:
            file_vector: Vector embedding of the file content
            correct_concept: The concept that IS correct
            file_name: Optional filename for logging
        """
        entry = {
            'vector': self._vector_to_list(file_vector),
            'concept': correct_concept,
            'file_name': file_name,
            'type': 'positive'
        }
        
        self._data['positives'].append(entry)
        self._save_store()
        
        print(f"[HardNegatives] Stored POSITIVE: '{file_name}' SHOULD match '{correct_concept}'")
    
    def get_confidence_adjustment(self, file_vector, concept: str) -> float:
        """
        Get confidence adjustment for a file-concept pair.
        
        Checks if the file vector is similar to any known negatives
        or positives for this concept.
        
        Args:
            file_vector: Vector embedding of the file
            concept: Concept being evaluated
            
        Returns:
            Adjustment value (negative reduces confidence, positive increases)
        """
        if file_vector is None:
            return 0.0
        
        file_vec_list = self._vector_to_list(file_vector)
        total_adjustment = 0.0
        
        # Check negatives
        for entry in self._data.get('negatives', []):
            if entry.get('concept', '').lower() != concept.lower():
                continue
            
            similarity = self._calculate_similarity(file_vec_list, entry.get('vector', []))
            
            if similarity >= self.SIMILARITY_THRESHOLD:
                total_adjustment += self.NEGATIVE_ADJUSTMENT
                print(f"[HardNegatives] Matched negative for '{concept}' (sim: {similarity:.3f})")
        
        # Check positives
        for entry in self._data.get('positives', []):
            if entry.get('concept', '').lower() != concept.lower():
                continue
            
            similarity = self._calculate_similarity(file_vec_list, entry.get('vector', []))
            
            if similarity >= self.SIMILARITY_THRESHOLD:
                total_adjustment += self.POSITIVE_ADJUSTMENT
                print(f"[HardNegatives] Matched positive for '{concept}' (sim: {similarity:.3f})")
        
        return total_adjustment
    
    def get_stats(self) -> Dict:
        """Get statistics about stored associations."""
        return {
            'total_negatives': len(self._data.get('negatives', [])),
            'total_positives': len(self._data.get('positives', [])),
            'concepts_with_negatives': list(set(
                e.get('concept') for e in self._data.get('negatives', [])
            )),
            'concepts_with_positives': list(set(
                e.get('concept') for e in self._data.get('positives', [])
            ))
        }
    
    def clear(self) -> None:
        """Clear all stored associations."""
        self._data = {'negatives': [], 'positives': []}
        self._save_store()


# Global instance for easy access
_hard_negatives = None

def get_hard_negatives() -> HardNegatives:
    """Get the global hard negatives instance."""
    global _hard_negatives
    if _hard_negatives is None:
        _hard_negatives = HardNegatives()
    return _hard_negatives


if __name__ == "__main__":
    # Quick test
    store = HardNegatives()
    stats = store.get_stats()
    print(f"[HardNegatives] Stats: {stats}")
    print(f"[HardNegatives] Store file: {store.store_path}")
