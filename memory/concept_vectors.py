"""
Concept Vectors - Learnable Concept Centroids
==============================================

Stores concept centroid vectors that can be permanently adjusted
through the dream consolidation process.

When the system repeatedly misclassifies files, the dream cycle
nudges the concept vector away from the problematic file vectors.

Storage: workspace/concept_vectors.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List


class ConceptVectors:
    """
    Stores learnable concept centroid vectors.
    
    These vectors start as the raw encoding of the concept name,
    but are adjusted over time based on user corrections.
    """
    
    # Default delta for nudging vectors
    DEFAULT_NUDGE_DELTA = 0.04
    
    def __init__(self, store_path: str = None):
        """
        Initialize the concept vectors store.
        
        Args:
            store_path: Path to store file. Defaults to workspace/concept_vectors.json
        """
        if store_path is None:
            project_root = Path(__file__).parent.parent
            store_dir = project_root / "workspace"
            store_dir.mkdir(exist_ok=True)
            self.store_path = store_dir / "concept_vectors.json"
        else:
            self.store_path = Path(store_path)
        
        self._data = self._load_store()
        self._consolidation_count = 0
    
    def _load_store(self) -> Dict:
        """Load store from disk."""
        if self.store_path.exists():
            try:
                with open(self.store_path, 'r') as f:
                    data = json.load(f)
                    # Ensure required keys exist
                    if 'concepts' not in data:
                        data['concepts'] = {}
                    if 'meta' not in data:
                        data['meta'] = {'total_adjustments': 0}
                    return data
            except (json.JSONDecodeError, IOError):
                return {'concepts': {}, 'meta': {'total_adjustments': 0}}
        return {'concepts': {}, 'meta': {'total_adjustments': 0}}
    
    def _save_store(self) -> None:
        """Save store to disk."""
        try:
            with open(self.store_path, 'w') as f:
                json.dump(self._data, f, indent=2)
        except IOError as e:
            print(f"[ConceptVectors] Warning: Could not save store: {e}")
    
    def _vector_to_list(self, vector) -> List[float]:
        """Convert numpy array to list for JSON storage."""
        if hasattr(vector, 'tolist'):
            return vector.tolist()
        return list(vector)
    
    def get_centroid(self, concept: str) -> Optional[List[float]]:
        """
        Get the centroid vector for a concept.
        
        Args:
            concept: The concept name
            
        Returns:
            Vector as list, or None if not stored
        """
        return self._data['concepts'].get(concept.lower())
    
    def set_centroid(self, concept: str, vector) -> None:
        """
        Set the centroid vector for a concept.
        
        Args:
            concept: The concept name
            vector: The vector (numpy array or list)
        """
        self._data['concepts'][concept.lower()] = self._vector_to_list(vector)
        self._save_store()
        print(f"[ConceptVectors] Set centroid for '{concept}'")
    
    def nudge_away(self, concept: str, file_vector, 
                   delta: float = None) -> Optional[List[float]]:
        """
        Nudge a concept vector AWAY from a file vector.
        
        Used when a file was wrongly classified as this concept.
        
        Args:
            concept: The concept to nudge
            file_vector: The file vector to move away from
            delta: How much to nudge (default 0.04)
            
        Returns:
            The new centroid vector, or None if concept not found
        """
        if delta is None:
            delta = self.DEFAULT_NUDGE_DELTA
        
        centroid = self.get_centroid(concept)
        if centroid is None:
            print(f"[ConceptVectors] Warning: No centroid for '{concept}', cannot nudge")
            return None
        
        # Convert to numpy for math
        c = np.array(centroid)
        f = np.array(self._vector_to_list(file_vector))
        
        # Direction from centroid toward file
        direction = f - c
        norm = np.linalg.norm(direction)
        
        if norm > 0:
            # Normalize and move AWAY (subtract)
            unit_direction = direction / norm
            new_centroid = c - (delta * unit_direction)
            
            # Re-normalize to unit length
            new_norm = np.linalg.norm(new_centroid)
            if new_norm > 0:
                new_centroid = new_centroid / new_norm
            
            self._data['concepts'][concept.lower()] = new_centroid.tolist()
            self._data['meta']['total_adjustments'] += 1
            self._consolidation_count += 1
            self._save_store()
            
            print(f"[ConceptVectors] Nudged '{concept}' AWAY by {delta:.4f}")
            return new_centroid.tolist()
        
        return centroid
    
    def nudge_toward(self, concept: str, file_vector,
                     delta: float = None) -> Optional[List[float]]:
        """
        Nudge a concept vector TOWARD a file vector.
        
        Used when a file IS correctly classified as this concept.
        
        Args:
            concept: The concept to nudge
            file_vector: The file vector to move toward
            delta: How much to nudge (default 0.04)
            
        Returns:
            The new centroid vector, or None if concept not found
        """
        if delta is None:
            delta = self.DEFAULT_NUDGE_DELTA
        
        centroid = self.get_centroid(concept)
        if centroid is None:
            # Initialize with file vector if no centroid exists
            self.set_centroid(concept, file_vector)
            print(f"[ConceptVectors] Initialized '{concept}' centroid from file vector")
            return self._vector_to_list(file_vector)
        
        # Convert to numpy for math
        c = np.array(centroid)
        f = np.array(self._vector_to_list(file_vector))
        
        # Direction from centroid toward file
        direction = f - c
        norm = np.linalg.norm(direction)
        
        if norm > 0:
            # Normalize and move TOWARD (add)
            unit_direction = direction / norm
            new_centroid = c + (delta * unit_direction)
            
            # Re-normalize to unit length
            new_norm = np.linalg.norm(new_centroid)
            if new_norm > 0:
                new_centroid = new_centroid / new_norm
            
            self._data['concepts'][concept.lower()] = new_centroid.tolist()
            self._data['meta']['total_adjustments'] += 1
            self._consolidation_count += 1
            self._save_store()
            
            print(f"[ConceptVectors] Nudged '{concept}' TOWARD by {delta:.4f}")
            return new_centroid.tolist()
        
        return centroid
    
    def get_stats(self) -> Dict:
        """Get statistics about stored concept vectors."""
        return {
            'total_concepts': len(self._data.get('concepts', {})),
            'total_adjustments': self._data.get('meta', {}).get('total_adjustments', 0),
            'concepts': list(self._data.get('concepts', {}).keys()),
            'session_consolidations': self._consolidation_count
        }
    
    def clear(self) -> None:
        """Clear all stored concept vectors."""
        self._data = {'concepts': {}, 'meta': {'total_adjustments': 0}}
        self._save_store()


# Global instance for easy access
_concept_vectors = None

def get_concept_vectors() -> ConceptVectors:
    """Get the global concept vectors instance."""
    global _concept_vectors
    if _concept_vectors is None:
        _concept_vectors = ConceptVectors()
    return _concept_vectors


if __name__ == "__main__":
    # Quick test
    store = ConceptVectors()
    stats = store.get_stats()
    print(f"[ConceptVectors] Stats: {stats}")
    print(f"[ConceptVectors] Store file: {store.store_path}")
