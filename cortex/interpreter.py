import yaml
import os

class LogosInterpreter:
    def __init__(self):
        pass

    def parse(self, file_path):
        """
        Parses a .logos file and returns the context.
        """
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} not found.")
            return None

        try:
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)
                return data
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML: {exc}")
            return None
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def interpret(self, data):
        """
        Interprets the parsed data and prints the context.
        """
        if not data:
            return

        context = data.get('Context', {})
        manifest = data.get('Manifest', 'Unknown')
        
        print(f"--- Logos Interpreter ---")
        print(f"Manifest: {manifest}")
        
        if context:
            gravity = context.get('Gravity', 'Normal')
            goal = context.get('Goal', 'No goal defined')
            print(f"System acknowledges: Gravity is {gravity.upper()}. Goal is {goal.upper()}.")
        else:
            print("No Context defined.")
            
        # Here we could process Rules and Constraints in the future
        print("-------------------------")
