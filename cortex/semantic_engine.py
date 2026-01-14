import os
from memory.vector_memory import VectorMemory

class SemanticEngine:
    def __init__(self):
        self.memory = VectorMemory()

    def evaluate(self, file_path, rules):
        """
        Evaluates a file against semantic rules.
        """
        if not rules:
            return None

        filename = os.path.basename(file_path)
        # We clean the filename to be more "natural language" friendly
        # e.g. "payment_confirmation_001.pdf" -> "payment confirmation 001"
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
                threshold = trigger.get('Threshold', 0.7) # Default threshold
                
                print(f"[SemanticEngine] Comparing against concept: '{concept}'...")
                
                concept_vector = self.memory.encode(concept)
                similarity = self.memory.calculate_similarity(file_vector, concept_vector)
                
                print(f"[SemanticEngine] Similarity: {similarity:.4f} (Threshold: {threshold})")
                
                if similarity >= threshold:
                    print(f"[SemanticEngine] MATCH FOUND! Concept: {concept}")
                    return rule.get('Action')

        return None
