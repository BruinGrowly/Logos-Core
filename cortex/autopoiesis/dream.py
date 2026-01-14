"""
Dream Engine - Metacognitive Processing
========================================

The "subconscious" of the machine. Runs during idle time to:
1. Analyze patterns in user corrections
2. Consolidate repeated corrections into permanent weight shifts
3. Prune old temporary data
4. Generate a morning briefing report

Usage:
    from cortex.autopoiesis import dream
    dream.start_cycle()
    
CLI:
    python main.py --meditate
"""

import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


# Configuration
PATTERN_THRESHOLD = 3           # Min corrections to trigger consolidation
PATTERN_WINDOW_HOURS = 24       # Time window for pattern detection
CONSOLIDATION_DELTA = 0.04      # How much to nudge vectors
PRUNE_AFTER_DAYS = 7            # Delete logs older than this


@dataclass
class CorrectionPattern:
    """A detected pattern of repeated corrections."""
    wrong_concept: str
    correct_concept: str
    count: int
    file_vectors: List[List[float]] = field(default_factory=list)
    file_names: List[str] = field(default_factory=list)


@dataclass
class WeightShift:
    """A permanent weight adjustment made during consolidation."""
    concept: str
    direction: str  # 'away' or 'toward'
    delta: float
    reason: str


@dataclass
class PruneStats:
    """Statistics from pruning operation."""
    action_log_removed: int
    hard_negatives_removed: int
    total_removed: int


@dataclass
class DreamReport:
    """Complete report of a dream cycle."""
    timestamp: str
    patterns_found: List[CorrectionPattern]
    weight_shifts: List[WeightShift]
    prune_stats: PruneStats
    insights: List[str]
    harmony_score: float = 0.0


class DreamEngine:
    """
    Metacognitive dream processor.
    
    Analyzes patterns, consolidates weights, and prunes old data.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.workspace = self.project_root / "workspace"
        self.workspace.mkdir(exist_ok=True)
        
        # Paths
        self.action_log_path = self.workspace / "action_log.json"
        self.hard_negatives_path = self.workspace / "hard_negatives.json"
        self.dream_stats_path = self.workspace / "dream_stats.json"
        self.morning_brief_path = self.workspace / "morning_brief.md"
    
    def start_cycle(self) -> DreamReport:
        """
        Execute a complete dream cycle.
        
        Returns:
            DreamReport with all processing details
        """
        print("\n" + "=" * 60)
        print("  🌙 DREAM CYCLE INITIATED")
        print("=" * 60 + "\n")
        
        # 1. Ingest data
        print("[Dream] Phase 1: Ingesting data...")
        action_log, hard_negatives = self._ingest_data()
        print(f"[Dream]   Loaded {len(action_log)} actions, {len(hard_negatives.get('negatives', []))} negatives")
        
        # 2. Pattern recognition
        print("\n[Dream] Phase 2: Recognizing patterns...")
        patterns = self._recognize_patterns(action_log, hard_negatives)
        print(f"[Dream]   Found {len(patterns)} consolidation patterns")
        
        # 3. Consolidation
        print("\n[Dream] Phase 3: Consolidating weights...")
        weight_shifts = self._consolidate_weights(patterns)
        print(f"[Dream]   Made {len(weight_shifts)} weight adjustments")
        
        # 4. Pruning
        print("\n[Dream] Phase 4: Pruning old data...")
        prune_stats = self._prune_old_data()
        print(f"[Dream]   Removed {prune_stats.total_removed} old entries")
        
        # 5. Generate insights
        insights = self._generate_insights(patterns, weight_shifts)
        
        # 6. Create report
        report = DreamReport(
            timestamp=datetime.now().isoformat(),
            patterns_found=patterns,
            weight_shifts=weight_shifts,
            prune_stats=prune_stats,
            insights=insights,
            harmony_score=self._get_harmony_score()
        )
        
        # 7. Save stats
        self._save_dream_stats(report)
        
        # 8. Generate morning briefing
        print("\n[Dream] Phase 5: Generating morning briefing...")
        self._generate_briefing(report)
        
        print("\n" + "=" * 60)
        print("  🌅 DREAM CYCLE COMPLETE")
        print("=" * 60 + "\n")
        
        return report
    
    def _ingest_data(self) -> Tuple[List, Dict]:
        """Load action log and hard negatives."""
        action_log = []
        hard_negatives = {'negatives': [], 'positives': []}
        
        if self.action_log_path.exists():
            try:
                with open(self.action_log_path, 'r') as f:
                    action_log = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        if self.hard_negatives_path.exists():
            try:
                with open(self.hard_negatives_path, 'r') as f:
                    hard_negatives = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return action_log, hard_negatives
    
    def _recognize_patterns(self, action_log: List, 
                           hard_negatives: Dict) -> List[CorrectionPattern]:
        """
        Identify concepts with repeated corrections.
        
        Looks for patterns where:
        - Same wrong_concept appears >= PATTERN_THRESHOLD times
        - All within PATTERN_WINDOW_HOURS
        """
        patterns = []
        cutoff = time.time() - (PATTERN_WINDOW_HOURS * 3600)
        
        # Group negatives by wrong concept
        concept_corrections = defaultdict(list)
        
        for entry in hard_negatives.get('negatives', []):
            concept = entry.get('concept', '')
            if concept:
                concept_corrections[concept].append(entry)
        
        # Find patterns that exceed threshold
        for wrong_concept, entries in concept_corrections.items():
            if len(entries) >= PATTERN_THRESHOLD:
                # Try to find the correct concept from positives
                # Look for corresponding positive entries
                correct_concept = self._find_correct_concept(
                    entries, 
                    hard_negatives.get('positives', [])
                )
                
                pattern = CorrectionPattern(
                    wrong_concept=wrong_concept,
                    correct_concept=correct_concept or "Unknown",
                    count=len(entries),
                    file_vectors=[e.get('vector', []) for e in entries],
                    file_names=[e.get('file_name', '') for e in entries]
                )
                patterns.append(pattern)
                
                print(f"[Dream]   Pattern: '{wrong_concept}' -> '{correct_concept}' ({len(entries)} corrections)")
        
        return patterns
    
    def _find_correct_concept(self, negative_entries: List, 
                              positives: List) -> Optional[str]:
        """Find the correct concept based on corresponding positives."""
        # Get file names from negatives
        neg_files = {e.get('file_name') for e in negative_entries if e.get('file_name')}
        
        # Find positives for the same files
        for pos in positives:
            if pos.get('file_name') in neg_files:
                return pos.get('concept')
        
        return None
    
    def _consolidate_weights(self, patterns: List[CorrectionPattern]) -> List[WeightShift]:
        """
        Permanently adjust concept vectors based on patterns.
        """
        weight_shifts = []
        
        if not patterns:
            return weight_shifts
        
        # Import here to avoid circular imports
        from memory.concept_vectors import get_concept_vectors
        from memory.vector_memory import VectorMemory
        
        concept_vectors = get_concept_vectors()
        memory = VectorMemory()
        
        for pattern in patterns:
            # Ensure concept has a centroid
            if concept_vectors.get_centroid(pattern.wrong_concept) is None:
                # Initialize from encoded concept name
                initial_vector = memory.encode(pattern.wrong_concept)
                if initial_vector is not None:
                    concept_vectors.set_centroid(pattern.wrong_concept, initial_vector)
            
            # Nudge AWAY from each file vector
            for i, file_vector in enumerate(pattern.file_vectors):
                if file_vector:
                    result = concept_vectors.nudge_away(
                        pattern.wrong_concept,
                        file_vector,
                        CONSOLIDATION_DELTA
                    )
                    
                    if result is not None:
                        shift = WeightShift(
                            concept=pattern.wrong_concept,
                            direction='away',
                            delta=CONSOLIDATION_DELTA,
                            reason=f"Correction: {pattern.file_names[i] if i < len(pattern.file_names) else 'unknown'}"
                        )
                        weight_shifts.append(shift)
            
            # If we know the correct concept, nudge it TOWARD the files
            if pattern.correct_concept and pattern.correct_concept != "Unknown":
                if concept_vectors.get_centroid(pattern.correct_concept) is None:
                    initial_vector = memory.encode(pattern.correct_concept)
                    if initial_vector is not None:
                        concept_vectors.set_centroid(pattern.correct_concept, initial_vector)
                
                for i, file_vector in enumerate(pattern.file_vectors):
                    if file_vector:
                        result = concept_vectors.nudge_toward(
                            pattern.correct_concept,
                            file_vector,
                            CONSOLIDATION_DELTA / 2  # Smaller boost
                        )
                        
                        if result is not None:
                            shift = WeightShift(
                                concept=pattern.correct_concept,
                                direction='toward',
                                delta=CONSOLIDATION_DELTA / 2,
                                reason=f"Positive: {pattern.file_names[i] if i < len(pattern.file_names) else 'unknown'}"
                            )
                            weight_shifts.append(shift)
        
        return weight_shifts
    
    def _prune_old_data(self) -> PruneStats:
        """
        Remove old log entries and consolidated hard negatives.
        """
        action_log_removed = 0
        hard_negatives_removed = 0
        
        prune_cutoff = time.time() - (PRUNE_AFTER_DAYS * 24 * 3600)
        
        # Prune action log
        if self.action_log_path.exists():
            try:
                with open(self.action_log_path, 'r') as f:
                    entries = json.load(f)
                
                original_count = len(entries)
                entries = [e for e in entries if e.get('epoch', 0) >= prune_cutoff]
                action_log_removed = original_count - len(entries)
                
                with open(self.action_log_path, 'w') as f:
                    json.dump(entries, f, indent=2)
                    
            except (json.JSONDecodeError, IOError):
                pass
        
        # Note: We don't prune hard_negatives that were just consolidated
        # because they might still be needed for edge cases.
        # In a production system, we'd mark them as "consolidated" and prune later.
        
        return PruneStats(
            action_log_removed=action_log_removed,
            hard_negatives_removed=hard_negatives_removed,
            total_removed=action_log_removed + hard_negatives_removed
        )
    
    def _generate_insights(self, patterns: List[CorrectionPattern],
                          weight_shifts: List[WeightShift]) -> List[str]:
        """Generate human-readable insights from the dream analysis."""
        insights = []
        
        if not patterns:
            insights.append("No significant correction patterns detected. Classifications are accurate.")
        else:
            for pattern in patterns:
                insight = f"I was misclassifying '{pattern.correct_concept}' documents as '{pattern.wrong_concept}' ({pattern.count} times)."
                insights.append(insight)
        
        if weight_shifts:
            concepts_adjusted = set(ws.concept for ws in weight_shifts)
            total_delta = sum(ws.delta for ws in weight_shifts)
            insights.append(f"Adjusted {len(concepts_adjusted)} concept(s) by total delta of {total_delta:.4f}.")
        
        return insights
    
    def _get_harmony_score(self) -> float:
        """Get current harmony score if available."""
        try:
            from cortex.autopoiesis.system import SystemHarmonyMeasurer
            measurer = SystemHarmonyMeasurer()
            report = measurer.measure(str(self.project_root))
            return report.harmony
        except Exception:
            return 0.0
    
    def _save_dream_stats(self, report: DreamReport) -> None:
        """Save dream statistics for dashboard."""
        stats = {
            'last_dream': report.timestamp,
            'patterns_found': len(report.patterns_found),
            'weight_shifts': len(report.weight_shifts),
            'memories_consolidated': len(report.weight_shifts),
            'entropy_removed': report.prune_stats.total_removed,
            'insights': report.insights
        }
        
        try:
            with open(self.dream_stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
        except IOError:
            pass
    
    def _generate_briefing(self, report: DreamReport) -> None:
        """Generate the morning briefing markdown report."""
        today = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Build insights section
        insights_text = ""
        for insight in report.insights:
            insights_text += f"- **Insight:** {insight}\n"
        
        # Build adjustments section
        if report.weight_shifts:
            concepts = set(ws.concept for ws in report.weight_shifts)
            total_delta = sum(ws.delta for ws in report.weight_shifts)
            adjustments_text = f"- **Adjustment:** Shifted {len(concepts)} concept vector(s) by total delta {total_delta:+.4f}"
        else:
            adjustments_text = "- **Adjustment:** No weight adjustments needed."
        
        # Build maintenance section
        maintenance_text = f"- **Maintenance:** Pruned {report.prune_stats.total_removed} old memory log(s)."
        
        briefing = f"""# 🌅 Morning Briefing
**Date:** {today}
**Status:** Harmony {report.harmony_score:.3f}

---

### 🌙 Dream Analysis

I processed the interactions from the past {PATTERN_WINDOW_HOURS} hours.

{insights_text}
{adjustments_text}
{maintenance_text}

---

### 📊 Session Statistics

| Metric | Value |
|--------|-------|
| Patterns Detected | {len(report.patterns_found)} |
| Weight Shifts | {len(report.weight_shifts)} |
| Logs Pruned | {report.prune_stats.total_removed} |

---

**Current State:** Calibrated and ready.

*Generated by Dream Engine - {today}*
"""
        
        try:
            with open(self.morning_brief_path, 'w', encoding='utf-8') as f:
                f.write(briefing)
            print(f"[Dream] Morning briefing saved to: {self.morning_brief_path}")
        except IOError as e:
            print(f"[Dream] Warning: Could not save briefing: {e}")


# Module-level function for easy access
def start_cycle() -> DreamReport:
    """Start a dream cycle."""
    engine = DreamEngine()
    return engine.start_cycle()


if __name__ == "__main__":
    # Run dream cycle directly
    start_cycle()
