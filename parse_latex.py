import os
import re
import json

def extract_braced_content(text, start_pos):
    """
    Finds the first '{' after start_pos and returns the content inside 
    the matching '}' along with the index right after the closing '}'.
    Handles nested braces correctly.
    """
    brace_start = text.find('{', start_pos)
    if brace_start == -1:
        return None, start_pos
    
    depth = 0
    for i in range(brace_start, len(text)):
        char = text[i]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                return text[brace_start+1 : i], i + 1
    return None, start_pos

def clean_tex_formatting(text):
    """
    Cleans LaTeX commands from structural titles/headers, keeping the plain text.
    """
    # Handle \texorpdfstring{first}{second} -> keep second
    if '\\texorpdfstring' in text:
        idx = text.find('\\texorpdfstring')
        first_group, next_idx = extract_braced_content(text, idx)
        if first_group is not None:
            second_group, _ = extract_braced_content(text, next_idx)
            if second_group is not None:
                text = text[:idx] + second_group + text[next_idx:]
    
    # Remove \index{...}
    text = re.sub(r'\\index\{[^{}]+\}', '', text)
    # Remove formatting command names but keep their braced content, e.g. \textbf{A} -> A
    text = re.sub(r'\\[a-zA-Z]+\{([^{}]+)\}', r'\1', text)
    # Remove commands without braces, e.g. \leavevmode -> ''
    text = re.sub(r'\\[a-zA-Z]+', '', text)
    # Clean remaining braces
    text = text.replace('{', '').replace('}', '').strip()
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_comments(text):
    """
    Removes LaTeX comments from the source text while preserving lines and escaping.
    """
    result = []
    for line in text.splitlines():
        # Entirely commented lines
        if line.strip().startswith('%'):
            continue
        
        # Find first unescaped %
        idx = 0
        in_escape = False
        comment_start = -1
        while idx < len(line):
            char = line[idx]
            if char == '\\':
                in_escape = not in_escape
            elif char == '%':
                if not in_escape:
                    comment_start = idx
                    break
                in_escape = False
            else:
                in_escape = False
            idx += 1
        
        if comment_start != -1:
            line = line[:comment_start]
        result.append(line)
    return '\n'.join(result)

def parse_exercises(block_text, chapter, section, subsection):
    """
    Parses an exercises block, splitting it into individual items and extracting
    questions and answers.
    """
    # Track list/environment depth to avoid splitting on nested \item commands
    pattern = re.compile(r'\\(begin|end|item|recommended|puzzle)\b')
    matches = list(pattern.finditer(block_text))
    
    match_info = []
    current_depth = 0
    
    for m in matches:
        cmd = m.group(1)
        start = m.start()
        end = m.end()
        
        if cmd == 'begin':
            env_name, next_idx = extract_braced_content(block_text, start)
            match_info.append({
                'cmd': 'begin',
                'name': env_name,
                'start': start,
                'end': next_idx if next_idx is not None else end,
                'depth_before': current_depth
            })
            current_depth += 1
        elif cmd == 'end':
            env_name, next_idx = extract_braced_content(block_text, start)
            current_depth -= 1
            match_info.append({
                'cmd': 'end',
                'name': env_name,
                'start': start,
                'end': next_idx if next_idx is not None else end,
                'depth_before': current_depth
            })
        else:
            match_info.append({
                'cmd': cmd,
                'start': start,
                'end': end,
                'depth_before': current_depth
            })
            
    # Group main list items (occurring at depth 0)
    item_starts = []
    i = 0
    while i < len(match_info):
        info = match_info[i]
        if info['depth_before'] == 0:
            if info['cmd'] in ('recommended', 'puzzle', 'item'):
                rec = False
                puz = False
                start_pos = info['start']
                
                curr_idx = i
                while curr_idx < len(match_info) and match_info[curr_idx]['cmd'] in ('recommended', 'puzzle'):
                    if match_info[curr_idx]['cmd'] == 'recommended':
                        rec = True
                    if match_info[curr_idx]['cmd'] == 'puzzle':
                        puz = True
                    curr_idx += 1
                
                if curr_idx < len(match_info) and match_info[curr_idx]['cmd'] == 'item':
                    item_starts.append({
                        'start': start_pos,
                        'item_cmd_end': match_info[curr_idx]['end'],
                        'recommended': rec,
                        'puzzle': puz
                    })
                    i = curr_idx + 1
                    continue
        i += 1

    exercises = []
    for idx_item, item_info in enumerate(item_starts):
        start_pos = item_info['start']
        end_pos = item_starts[idx_item + 1]['start'] if idx_item + 1 < len(item_starts) else len(block_text)
        
        item_text_block = block_text[start_pos:end_pos]
        item_matches = [m for m in match_info if m['start'] >= start_pos and m['end'] <= end_pos]
        
        ans_start = None
        ans_cmd_end = None
        ans_end = None
        
        # Find \begin{answer} ... \end{answer} at depth 0 relative to the item
        for m in item_matches:
            if m['cmd'] == 'begin' and m['name'] == 'answer' and m['depth_before'] == 0:
                ans_start = m['start'] - start_pos
                ans_cmd_end = m['end'] - start_pos
            elif m['cmd'] == 'end' and m['name'] == 'answer' and m['depth_before'] == 0:
                ans_end = m['start'] - start_pos
        
        if ans_start is not None and ans_end is not None:
            q_start = item_info['item_cmd_end'] - start_pos
            question_content = item_text_block[q_start : ans_start].strip()
            answer_content = item_text_block[ans_cmd_end : ans_end].strip()
        else:
            q_start = item_info['item_cmd_end'] - start_pos
            question_content = item_text_block[q_start:].strip()
            answer_content = ""
            
        label = None
        label_match = re.search(r'\\label\{([^{}]+)\}', question_content)
        if label_match:
            label = label_match.group(1).strip()
            
        exercises.append({
            'category': 'exercise',
            'id': f"{chapter}.{section}.{subsection}.{idx_item + 1}",
            'index': idx_item + 1,
            'label': label,
            'recommended': item_info['recommended'],
            'puzzle': item_info['puzzle'],
            'question': question_content,
            'answer': answer_content,
            'location': {
                'chapter': chapter,
                'section': section,
                'subsection': subsection
            }
        })
        
    return exercises

def parse_latex_file(filepath):
    """
    Main parser function that reads a latex file and extracts all structural sections,
    definitions, theorems, lemmas, corollaries, examples, and exercises.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_text = f.read()
        
    text = remove_comments(raw_text)
    
    idx = 0
    current_chapter = ""
    current_section = ""
    current_subsection = ""
    
    elements = []
    
    command_pattern = re.compile(
        r'\\(chapter|section|subsection|subsectionoptional|begin)\b'
    )
    
    while idx < len(text):
        match = command_pattern.search(text, idx)
        if not match:
            break
        
        cmd_type = match.group(1)
        start_pos = match.start()
        next_pos = match.end()
        
        if cmd_type in ('chapter', 'section', 'subsection', 'subsectionoptional'):
            content, next_idx = extract_braced_content(text, start_pos)
            if content is not None:
                cleaned = clean_tex_formatting(content)
                if cmd_type == 'chapter':
                    current_chapter = cleaned
                    current_section = ""
                    current_subsection = ""
                elif cmd_type == 'section':
                    current_section = cleaned
                    current_subsection = ""
                else:
                    current_subsection = cleaned
                idx = next_idx
            else:
                idx = next_pos
                
        elif cmd_type == 'begin':
            env_name, next_idx = extract_braced_content(text, start_pos)
            if env_name is not None:
                idx = next_idx
                if env_name in ('definition', 'theorem', 'lemma', 'corollary', 'example', 'remark'):
                    end_tag = f"\\end{{{env_name}}}"
                    end_idx = text.find(end_tag, idx)
                    if end_idx != -1:
                        block_content = text[idx : end_idx].strip()
                        
                        # Extract optional title in square brackets, e.g. \begin{theorem}[Gauss's Method]
                        title = None
                        title_match = re.match(r'^\s*\[([^\]]+)\]', text[idx:])
                        if title_match:
                            title = title_match.group(1).strip()
                            block_content = text[idx + title_match.end() : end_idx].strip()
                            
                        label = None
                        label_match = re.search(r'\\label\{([^{}]+)\}', block_content)
                        if label_match:
                            label = label_match.group(1).strip()
                            
                        elements.append({
                            'category': 'theory',
                            'type': env_name,
                            'title': title,
                            'label': label,
                            'content': block_content,
                            'location': {
                                'chapter': current_chapter,
                                'section': current_section,
                                'subsection': current_subsection
                            }
                        })
                        idx = end_idx + len(end_tag)
                    else:
                        idx = next_pos
                elif env_name == 'exercises':
                    end_tag = "\\end{exercises}"
                    end_idx = text.find(end_tag, idx)
                    if end_idx != -1:
                        exercises_block = text[idx : end_idx].strip()
                        parsed_exercises = parse_exercises(
                            exercises_block, 
                            current_chapter, 
                            current_section, 
                            current_subsection
                        )
                        elements.extend(parsed_exercises)
                        idx = end_idx + len(end_tag)
                    else:
                        idx = next_pos
                else:
                    # Ignore other begin environments (they are handled as nested content)
                    idx = next_pos
            else:
                idx = next_pos
        else:
            idx = next_pos
            
    return elements

if __name__ == '__main__':
    src_file = os.path.join("linear-algebra-master", "src", "gr", "gr1.tex")
    out_file = "parsed_gr1.json"
    
    print(f"Parsing LaTeX file: {src_file}...")
    try:
        parsed_elements = parse_latex_file(src_file)
        
        # Save to JSON
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(parsed_elements, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully saved {len(parsed_elements)} elements to {out_file}!")
        
        # Summary report
        counts = {}
        for elem in parsed_elements:
            cat = elem['category']
            t = elem.get('type', cat)
            counts[t] = counts.get(t, 0) + 1
            
        print("\nSummary of Extracted Elements:")
        for t, count in counts.items():
            print(f"  - {t.capitalize()}: {count}")
            
    except Exception as e:
        print(f"Error parsing file: {e}")
        import traceback
        traceback.print_exc()
