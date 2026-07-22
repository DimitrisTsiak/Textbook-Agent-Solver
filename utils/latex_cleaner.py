import os
import re

def clean_latex_for_embeddings(text):
    """
    Cleans LaTeX source code to optimize it for generating vector embeddings.
    Removes comments, indices, labels, formatting command names, and simplifies
    math blocks, matrices, systems of equations, and fractions into clean text.
    """
    if not text:
        return ""
    
    # 1. Clean basic spaces, non-breaking spaces, and custom dashes
    text = text.replace('~', ' ')
    text = text.replace('\\Dash', ' - ')
    text = text.replace('\\dash', ' - ')
    text = text.replace('\\spaceforemptycolumn', '')
    text = text.replace('\\suchthat', ' such that ')
    
    # 2. Simplify fractions: \frac{a}{b} -> a/b
    # Handle up to 3 levels of nested fractions
    for _ in range(3):
        text = re.sub(r'\\frac\{([^{}]+)\}\{([^{}]+)\}', r'\1/\2', text)
        
    # 3. Simplify column vectors: \colvec[r]{5 \\\\ -5} -> [5, -5]
    def clean_colvec(match):
        cells_text = match.group(1)
        # Replace row separators with commas
        cells_clean = re.sub(r'\\\\', ', ', cells_text)
        # Collapse multiple spaces
        cells_clean = re.sub(r'\s+', ' ', cells_clean).strip()
        return f"[{cells_clean}]"
    
    # Matches \colvec[option]{content} or \colvec{content}
    text = re.compile(r'\\colvec(?:\[[^\]]*\])?\{([^{}]+)\}').sub(clean_colvec, text)

    # 4. Remove labels, indices, and citations completely
    text = re.sub(r'\\label\{[^{}]+\}', '', text)
    text = re.sub(r'\\index\{[^{}]+\}', '', text)
    text = re.sub(r'\\cite\{[^{}]+\}', '', text)
    
    # 5. Clean up definition highlight formatting: \definend{word} -> word
    text = re.sub(r'\\definend\{([^{}]+)\}', r'\1', text)
    
    # 6. Simplify cross-references: \nearbytheorem{th:name} -> Theorem
    def clean_nearby(match):
        ref_type = match.group(1).lower()
        if ref_type == 'theorem':
            return 'Theorem'
        elif ref_type == 'definition':
            return 'Definition'
        elif ref_type == 'example':
            return 'Example'
        elif ref_type == 'lemma':
            return 'Lemma'
        elif ref_type == 'corollary':
            return 'Corollary'
        elif ref_type == 'remark':
            return 'Remark'
        elif ref_type == 'exercise':
            return 'Exercise'
        else:
            return ref_type.capitalize()
            
    text = re.compile(r'\\nearby([a-zA-Z]+)\{[^{}]+\}').sub(clean_nearby, text)

    # 7. Convert math environments (linsys, amat, align, equation) into comma-separated equations
    text = re.sub(r'\\\\', ', ', text)
    text = text.replace('&', ' ')
    
    # Remove \begin{env} and \end{env} tags, including optional arguments
    text = re.sub(r'\\begin\{[a-zA-Z]+\*?\}(?:\{[^{}]+\})?(?:\[[^\]]+\])?', '', text)
    text = re.sub(r'\\end\{[a-zA-Z]+\*?\}', '', text)
    
    # 8. Simplify custom reduction step labels, e.g. \grstep[-2\rho_1 + \rho_2] -> (-2\rho_1 + \rho_2)
    text = re.sub(r'\\grstep(?:\[[^\]]*\])?(?:\{([^{}]+)\})?', lambda m: f"({m.group(1)})" if m.group(1) else "", text)
    text = re.sub(r'\\repeatedgrstep\{([^{}]+)\}', r'(\1)', text)
    
    # 9. Strip out mathematical delimiters
    text = text.replace('\\(', '').replace('\\)', '')
    text = text.replace('\\[', '').replace('\\]', '')
    text = text.replace('$', '')
    
    # 10. Clean up formatting macros, e.g., \textbf{Text} -> Text
    for _ in range(3):
        text = re.sub(r'\\[a-zA-Z]+\{([^{}]+)\}', r'\1', text)
    
    # Remove remaining loose backslash commands (e.g. \leavevmode)
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    
    # 11. Normalize remaining curly braces and clean whitespaces
    text = text.replace('{', '').replace('}', '')
    text = re.sub(r'\s+', ' ', text)
    
    # Normalize spacing around commas
    text = re.sub(r'\s*,\s*', ', ', text)
    # Remove double commas
    text = re.sub(r',(\s*,)+', ',', text)
    
    # Remove leading/trailing commas and spaces
    text = text.strip().strip(',').strip()
    
    return text

if __name__ == '__main__':
    # Test cases to verify cleaner output
    test_latex = r"""\label{df:EchelonForm}
In each row of a system, \definend{leading variable}\index{echelon form!leading variable}.
A system is in echelon form if each leading variable is to the right.
\begin{linsys}{2}
  x  &+  &3y  &=  &1  \\
 2x  &+  &y   &=  &-3
\end{linsys}
We use \nearbytheorem{th:GaussMethod} with \colvec[r]{5 \\\\ -5} and \frac{1}{2}x."""
    
    print("Raw LaTeX:")
    print(test_latex)
    print("\nCleaned output:")
    print(clean_latex_for_embeddings(test_latex))
