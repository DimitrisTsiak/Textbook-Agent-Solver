import io
import sys
import contextlib
import traceback
import ast
from utils.tracing import observe, update_current_observation

def is_safe_code(code: str) -> tuple[bool, str]:
    """
    Statically analyzes python code using AST to check for forbidden statements
    or sandbox escape attempts.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"Syntax Error: {str(e)}"
        
    for node in ast.walk(tree):
        # 1. Block all imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "Security Error: Import statements are forbidden. All required math libraries (SymPy, NumPy) are pre-imported."
            
        # 2. Block sensitive built-ins
        if isinstance(node, ast.Name):
            forbidden_builtins = {'open', 'eval', 'exec', '__import__', 'compile', 'globals', 'locals', 'input'}
            if node.id in forbidden_builtins:
                return False, f"Security Error: Use of forbidden built-in '{node.id}' is blocked."
                
        # 3. Block double-underscore access (dunder attributes)
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('__'):
                return False, f"Security Error: Accessing private attribute '{node.attr}' is blocked."
                
    return True, ""

@observe(as_type="tool", name="calculator-tool")
def calculate_linear_algebra(code: str) -> str:
    """
    Execute Python code to perform mathematical calculations using SymPy and NumPy.
    The code should print the final results to stdout.
    
    SymPy is pre-imported (both as 'sp' and via 'from sympy import *').
    NumPy is pre-imported as 'np'.
    
    Args:
        code: A string containing Python code to execute.
        
    Returns:
        The printed stdout of the execution, or the error message if code is unsafe or fails.
    """
    print(f"\n[Tool Execution] calculate_linear_algebra called.")
    update_current_observation(input={"code": code})
    
    # 1. Statically sanitize code using AST
    is_safe, error_msg = is_safe_code(code)
    if not is_safe:
        print(f"  [Security Check] Failed: {error_msg}")
        update_current_observation(output=error_msg, metadata={"security_check_passed": False})
        return error_msg
        
    # Setup execution environment with numpy and sympy pre-imported
    local_env = {}
    
    # Restrict builtins to safe mathematical/printing/iteration functions
    safe_builtins = {
        'print': print,
        'range': range,
        'len': len,
        'int': int,
        'float': float,
        'list': list,
        'dict': dict,
        'set': set,
        'str': str,
        'enumerate': enumerate,
        'zip': zip,
        'sum': sum,
        'max': max,
        'min': min,
        'abs': abs,
        'round': round,
        'pow': pow,
        'divmod': divmod,
        'map': map,
        'filter': filter,
        'any': any,
        'all': all,
        'bool': bool,
        'tuple': tuple,
        'type': type,
    }
    
    global_env = {
        '__builtins__': safe_builtins
    }
    
    try:
        import sympy as sp
        global_env['sp'] = sp
        # Import all public symbols of sympy into the global env
        for name in dir(sp):
            if not name.startswith('_'):
                global_env[name] = getattr(sp, name)
    except ImportError:
        return "Error: sympy library is not installed. Please install it using 'pip install sympy'."
        
    try:
        import numpy as np
        global_env['np'] = np
    except ImportError:
        pass # numpy is optional but useful
        
    # Capture stdout using thread-safe redirect
    redirected_output = io.StringIO()
    
    try:
        # redirect_stdout scopes the capture to this call, preventing
        # output bleed between concurrent requests in FastAPI's thread pool.
        with contextlib.redirect_stdout(redirected_output):
            exec(code, global_env, local_env)
        output = redirected_output.getvalue()
        final_res = output if output.strip() else "Code executed successfully, but returned no stdout. Did you forget to print() your results?"
        update_current_observation(output=final_res)
        return final_res
    except Exception as e:
        # Capture traceback on failure to help model self-correct
        tb = traceback.format_exc()
        err_res = f"Error executing code:\n{tb}"
        update_current_observation(output=err_res)
        return err_res


if __name__ == '__main__':
    # Test cases to verify sanitizer and execution
    print("Testing calculate_linear_algebra tool...")
    
    # Case A: Safe math code
    safe_code = """
A = Matrix([[2, 3], [1, -1]])
b = Matrix([13, -1])
sol = A.LUsolve(b)
print(f"Solution: x = {sol[0]}, y = {sol[1]}")
"""
    print("\nExecuting Safe Code:")
    print(calculate_linear_algebra(safe_code))
    
    # Case B: Malicious code trying to import os
    unsafe_code = """
import os
os.system("echo 'hack'")
"""
    print("\nExecuting Unsafe Code:")
    print(calculate_linear_algebra(unsafe_code))
