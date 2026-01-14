"""
Semantic Engine - Deep Scan Cascade
====================================

Evaluates files against semantic rules using a two-stage cascade:

1. Surface Scan: Analyze filename similarity (fast)
2. Deep Scan: Read file content if filename score is ambiguous (score < 0.65)

The Deep Scan uses the reader module to extract content and the
sensory cache to avoid re-vectorizing the same files.
"""

import os
from memory.vector_memory import VectorMemory
from memory.sensory_cache import SensoryCache
from memory.hard_negatives import get_hard_negatives
from sensory.reader import read_header, is_supported


# Threshold below which we trigger a deep scan
DEEP_SCAN_THRESHOLD = 0.65


class SemanticEngine:
    """
    Semantic analysis engine with Deep Scan Cascade.
    
    Combines filename analysis with content extraction for
    more accurate semantic matching.
    """
    
    def __init__(self):
        self.memory = VectorMemory()
        self.cache = SensoryCache()
        self.hard_negatives = get_hard_negatives()
        
        # Metrics for dashboard
        self.surface_scan_count = 0
        self.deep_scan_count = 0
        self.rlhf_adjustments = 0  # Count of times RLHF affected a decision
    
    def evaluate(self, file_path, rules):
        """
        Evaluates a file against semantic rules using Deep Scan Cascade.
        
        Args:
            file_path: Path to the file to evaluate
            rules: List of rules with Trigger conditions and Actions
            
        Returns:
            Action dict if a rule matches, None otherwise
        """
        if not rules:
            return None

        filename = os.path.basename(file_path)
        # Clean the filename for natural language processing
        clean_name = filename.replace('_', ' ').replace('-', ' ')
        name_without_ext = os.path.splitext(clean_name)[0]

        print(f"[SemanticEngine] Perceiving '{name_without_ext}'...")
        
        # Encode the filename once
        file_vector = self.memory.encode(name_without_ext)
        if file_vector is None:
            return None

        for rule in rules:
            trigger = rule.get('Trigger', {})
            
            if trigger.get('Type') == 'Semantic_Match':
                concept = trigger.get('Concept')
                threshold = trigger.get('Threshold', 0.7)
                
                print(f"[SemanticEngine] Comparing against concept: '{concept}'...")
                
                concept_vector = self.memory.encode(concept)
                
                # ========== STAGE 1: SURFACE SCAN (Filename) ==========
                similarity = self.memory.calculate_similarity(file_vector, concept_vector)
                self.surface_scan_count += 1
                
                print(f"[SemanticEngine] Surface Scan: {similarity:.4f} (Threshold: {threshold})")
                
                # ========== STAGE 2: DEEP SCAN (Content) ==========
                # Trigger deep scan if surface score is ambiguous
                if similarity < DEEP_SCAN_THRESHOLD and is_supported(file_path):
                    print(f"[SemanticEngine] Surface score weak ({similarity:.3f} < {DEEP_SCAN_THRESHOLD}), initiating Deep Scan...")
                    
                    content_similarity = self._deep_scan(file_path, concept_vector)
                    
                    if content_similarity is not None:
                        self.deep_scan_count += 1
                        # Use the better of the two scores
                        old_similarity = similarity
                        similarity = max(similarity, content_similarity)
                        print(f"[SemanticEngine] Deep Scan: {content_similarity:.4f} (Combined: {similarity:.4f})")
                        
                        if content_similarity > old_similarity:
                            print(f"[SemanticEngine] Content revealed higher similarity! ({content_similarity:.4f} > {old_similarity:.4f})")
                
                # ========== STAGE 3: RLHF ADJUSTMENT ==========
                # Apply learned corrections from user feedback
                adjustment = self.hard_negatives.get_confidence_adjustment(file_vector, concept)
                if adjustment != 0:
                    old_sim = similarity
                    similarity = max(0.0, min(1.0, similarity + adjustment))  # Clamp to [0, 1]
                    self.rlhf_adjustments += 1
                    print(f"[SemanticEngine] RLHF Adjustment: {old_sim:.4f} -> {similarity:.4f} ({adjustment:+.2f})")
                
                # Final decision
                if similarity >= threshold:
                    print(f"[SemanticEngine] MATCH FOUND! Concept: {concept}")
                    # Add concept to action for logging
                    action = rule.get('Action', {}).copy()
                    action['Concept'] = concept
                    return action

        return None
    
    def _deep_scan(self, file_path, concept_vector):
        """
        Perform deep content scan on a file.
        
        Args:
            file_path: Path to file
            concept_vector: Vector to compare against
            
        Returns:
            Similarity score, or None if content unreadable
        """
        # Check cache first
        content_vector = self.cache.get_vector(file_path)
        
        if content_vector is None:
            # Cache miss - read and vectorize content
            content = read_header(file_path, limit=1000)
            
            if content is None or len(content.strip()) < 10:
                print(f"[SemanticEngine] Deep Scan: No readable content")
                return None
            
            print(f"[SemanticEngine] Deep Scan: Read {len(content)} characters")
            
            # Vectorize the content
            content_vector = self.memory.encode(content)
            
            if content_vector is not None:
                # Store in cache for future use
                self.cache.store_vector(file_path, content_vector)
        else:
            print(f"[SemanticEngine] Deep Scan: Cache HIT")
        
        if content_vector is None:
            return None
        
        # Calculate similarity with content vector
        return self.memory.calculate_similarity(content_vector, concept_vector)
    
    def get_sensory_depth(self) -> float:
        """
        Calculate the Sensory Depth metric.
        
        Returns:
            Percentage of decisions that used Deep Scan (0-100)
        """
        total = self.surface_scan_count
        if total == 0:
            return 0.0
        return (self.deep_scan_count / total) * 100
    
    def get_metrics(self) -> dict:
        """Get all engine metrics for dashboard."""
        cache_stats = self.cache.get_stats()
        hn_stats = self.hard_negatives.get_stats()
        return {
            'surface_scans': self.surface_scan_count,
            'deep_scans': self.deep_scan_count,
            'sensory_depth_percent': round(self.get_sensory_depth(), 1),
            'cache_hits': cache_stats['hits'],
            'cache_misses': cache_stats['misses'],
            'cache_hit_rate': cache_stats['hit_rate_percent'],
            'cached_files': cache_stats['cached_entries'],
            'rlhf_adjustments': self.rlhf_adjustments,
            'hard_negatives': hn_stats['total_negatives'],
            'positives': hn_stats['total_positives']
        }
    
    def reset_metrics(self):
        """Reset scan counters."""
        self.surface_scan_count = 0
        self.deep_scan_count = 0
        self.rlhf_adjustments = 0
    
    def save_metrics(self, workspace_path: str = None):
        """
        Save metrics to JSON file for dashboard consumption.
        
        Args:
            workspace_path: Path to workspace folder. Defaults to ./workspace
        """
        import json
        from pathlib import Path
        
        if workspace_path is None:
            workspace_path = Path(__file__).parent.parent / "workspace"
        else:
            workspace_path = Path(workspace_path)
        
        workspace_path.mkdir(exist_ok=True)
        metrics_file = workspace_path / "sensory_metrics.json"
        
        metrics = self.get_metrics()
        
        try:
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except IOError as e:
            print(f"[SemanticEngine] Warning: Could not save metrics: {e}")
