
"""
Module for in-context-based workflow generation in AutoFlow.
This module reads the current context from the Notebook and dynamically suggests or builds the next workflow steps.
"""

from typing import List, Dict, Any
from utils.notebook import Notebook



# Dynamic registry for workflow modules
WORKFLOW_MODULES: Dict[str, Dict[str, Any]] = {}

def register_workflow_module(name: str, description: str, requires: list, produces: list):
    """
    Register a new workflow module at runtime.
    """
    WORKFLOW_MODULES[name] = {
        'description': description,
        'requires': requires,
        'produces': produces
    }

def unregister_workflow_module(name: str):
    """
    Remove a workflow module from the registry.
    """
    if name in WORKFLOW_MODULES:
        del WORKFLOW_MODULES[name]

def list_workflow_modules() -> Dict[str, Dict[str, Any]]:
    """
    Return the current workflow module registry.
    """
    return WORKFLOW_MODULES



def get_contextual_workflow(notebook: Notebook) -> List[str]:
    """
    Dynamically suggests the next workflow steps based on the current context in the notebook
    and the requirements of available modules.
    Returns a list of suggested module names.
    """
    context_keys = set(notebook.list_keys())
    suggestions = []
    for module_name, module_info in WORKFLOW_MODULES.items():
        missing_outputs = [out for out in module_info['produces'] if out not in context_keys]
        requirements_met = all(req in context_keys for req in module_info['requires'])
        if missing_outputs and requirements_met:
            suggestions.append(module_name)
    return suggestions


# New function for interactive workflow selection and registration
def interactive_workflow_selection(notebook: Notebook, prompt_func=input):
    """
    Show suggested workflow modules, prompt user to select, and register if missing.
    Returns the selected module name.
    """
    suggestions = get_contextual_workflow(notebook)
    if not suggestions:
        print("No suggested workflow modules available.")
        return None
    print("Suggested workflow modules:")
    for idx, name in enumerate(suggestions, 1):
        print(f"  {idx}. {name}")
    while True:
        selection = prompt_func(f"Select a module by number or name: ").strip()
        if selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(suggestions):
                selected = suggestions[idx]
                break
            else:
                print("Invalid number. Try again.")
        elif selection in suggestions:
            selected = selection
            break
        else:
            print("Invalid selection. Try again.")
    # If the selected module is not registered, prompt for details
    if selected not in WORKFLOW_MODULES:
        auto_register_module_if_missing(selected, prompt_func=prompt_func)
    return selected

# Example function to execute a suggested workflow step (stub)
def execute_workflow_step(notebook: Notebook, step: str, params: Dict[str, Any] = None):
    """
    Executes a workflow step and updates the notebook. This is a stub for integration.
    """
    # Here you would call the actual module/function for the step
    # For now, just record that the step was executed
    notebook.write(step, input_data=None, short_description=f"Executed {step}")
    return f"Step '{step}' executed."



# Function to load workflow modules from a JSON config file
import json
import os

def load_workflow_modules_from_json(json_path: str):
    """
    Load workflow modules from a JSON file and register them dynamically.
    The JSON file should be a list of module definitions, each with 'name', 'description', 'requires', 'produces'.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Config file not found: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        modules = json.load(f)
    for module in modules:
        register_workflow_module(
            name=module['name'],
            description=module.get('description', ''),
            requires=module.get('requires', []),
            produces=module.get('produces', [])
        )

def auto_register_module_if_missing(module_name: str, prompt_func=input):
    """
    If a module is missing, prompt the user for details and register it dynamically.
    prompt_func is used for input (can be replaced for testing).
    """
    if module_name in WORKFLOW_MODULES:
        return False  # Already exists
    print(f"Module '{module_name}' not found. Let's create it!")
    description = prompt_func(f"Enter a short description for '{module_name}': ")
    requires = prompt_func(f"Enter required inputs (comma-separated): ").split(',')
    requires = [r.strip() for r in requires if r.strip()]
    produces = prompt_func(f"Enter outputs produced (comma-separated): ").split(',')
    produces = [p.strip() for p in produces if p.strip()]
    register_workflow_module(module_name, description, requires, produces)
    print(f"Module '{module_name}' registered!")
    return True
