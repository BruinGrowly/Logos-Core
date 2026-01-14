import os

class RulesEngine:
    def __init__(self):
        pass

    def evaluate(self, file_path, rules):
        """
        Evaluates a file against a list of rules and returns the first matching action.
        """
        if not rules:
            return None

        filename = os.path.basename(file_path).lower()
        _, extension = os.path.splitext(filename)

        print(f"[RulesEngine] Evaluating {filename} against {len(rules)} rules...")

        for rule in rules:
            trigger = rule.get('Trigger', {})
            action = rule.get('Action', {})
            
            if self._check_trigger(trigger, filename, extension):
                print(f"[RulesEngine] MATCH FOUND! Trigger: {trigger}")
                return action
        
        return None

    def _check_trigger(self, trigger, filename, extension):
        """
        Checks if a file matches a specific trigger condition.
        """
        t_type = trigger.get('Type')
        if t_type != 'File_Event':
            return False

        condition = trigger.get('Condition')
        
        if condition == 'Filename':
            keywords = trigger.get('Contains', [])
            # If keywords is a string, make it a list
            if isinstance(keywords, str):
                keywords = [keywords]
            
            for keyword in keywords:
                if keyword.lower() in filename:
                    return True
                    
        elif condition == 'Extension':
            allowed_exts = trigger.get('Is', [])
            if isinstance(allowed_exts, str):
                allowed_exts = [allowed_exts]
                
            # Normalize extensions (ensure they start with .)
            allowed_exts = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in allowed_exts]
            
            if extension in allowed_exts:
                return True

        return False
