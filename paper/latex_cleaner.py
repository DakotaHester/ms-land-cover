#!/usr/bin/env python3

"""
LaTeX Project Cleaner

This script parses a main LaTeX file to find all its dependencies
(e.g., \input, \includegraphics, \bibliography) and copies only the
necessary files to a clean output directory, retaining the project structure.

It also cleans .tex and .bib files by:
1. Removing all full-line and inline LaTeX comments (respecting escaped \%).
2. Squeezing multiple blank lines (2 or more) into a single blank line.

Usage:
    python latex_cleaner.py /path/to/your/main.tex /path/to/clean_output_directory

Example:
    python latex_cleaner.py ./my_project/main.tex ./my_project_clean
"""

import os
import re
import shutil
import argparse
import logging

# --- Regex for Parsing LaTeX Dependencies ---

# \input{...} or \include{...}
# We capture the path inside the braces
RE_INPUT = re.compile(r'\\(?:input|include)\{([^}]+)\}')

# \bibliography{...}
# We capture the comma-separated list of bib files
RE_BIBLIOGRAPHY = re.compile(r'\\bibliography\{([^}]+)\}')

# \includegraphics[...]{...}
# We only capture the path from the mandatory braces
RE_GRAPHICSPATH = re.compile(r'\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}')

# \usepackage[...]{...} or \documentclass[...]{...}
# Used to find local .sty or .cls files
RE_PACKAGE = re.compile(r'\\usepackage(?:\[[^\]]+\])?\{([^}]+)\}')
RE_CLASS = re.compile(r'\\documentclass(?:\[[^\]]+\])?\{([^}]+)\}')

# Regex to find a non-escaped '%' and remove it and the rest of the line
RE_COMMENT = re.compile(r'(?<!\\)%.*')

# Regex to squeeze 3 or more newlines (2+ blank lines) into 2 newlines (1 blank line)
RE_SQUEEZE_NEWLINES = re.compile(r'\n{3,}')


def clean_tex_content(content: str) -> str:
    """
    Cleans the content of a .tex or .bib file.
    
    - Removes all inline and full-line comments (respecting \%).
    - Squeezes multiple blank lines into one.
    """
    cleaned_lines = []
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('%'):
            continue  # Skip full-line comments entirely
        cleaned_line = RE_COMMENT.sub('', line).rstrip()
        if cleaned_line.strip() != '':
            cleaned_lines.append(cleaned_line)
        else:
            # Only add a blank line if the previous line wasn't blank
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
    cleaned_content = '\n'.join(cleaned_lines)
    cleaned_content = RE_SQUEEZE_NEWLINES.sub('\n\n', cleaned_content)
    return cleaned_content.strip()


def find_dependencies(content: str, project_root: str) -> set:
    """
    Parses .tex content to find all file dependencies.
    Returns a set of file paths relative to the project root.
    """
    dependencies = set()

    # Find \input and \include
    for match in RE_INPUT.finditer(content):
        dep_file = match.group(1)
        if not dep_file.endswith('.tex'):
            dep_file += '.tex'
        dependencies.add(dep_file)

    # Find \bibliography
    for match in RE_BIBLIOGRAPHY.finditer(content):
        # Can be a comma-separated list
        bib_files = match.group(1).split(',')
        for bib_file in bib_files:
            bib_file = bib_file.strip()
            if not bib_file.endswith('.bib'):
                bib_file += '.bib'
            dependencies.add(bib_file)

    # Find \includegraphics
    for match in RE_GRAPHICSPATH.finditer(content):
        # graphicx can auto-find extensions, so we don't add them.
        # We also don't parse \graphicspath{}, assuming standard layout.
        dependencies.add(match.group(1))

    # Find local \usepackage
    for match in RE_PACKAGE.finditer(content):
        package_files = match.group(1).split(',')
        for pkg_file in package_files:
            pkg_file = pkg_file.strip()
            if not pkg_file.endswith('.sty'):
                pkg_file += '.sty'
            # Only add if it's a local file, not a system package
            if os.path.exists(os.path.join(project_root, pkg_file)):
                dependencies.add(pkg_file)

    # Find local \documentclass
    for match in RE_CLASS.finditer(content):
        class_file = match.group(1).strip()
        if not class_file.endswith('.cls'):
            class_file += '.cls'
        # Only add if it's a local file
        if os.path.exists(os.path.join(project_root, class_file)):
            dependencies.add(class_file)

    return dependencies

def strip_comments_for_dependency_parsing(content: str) -> str:
    lines = []
    for line in content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('%'):
            continue
        # Remove inline comments
        line_no_comment = RE_COMMENT.sub('', line)
        lines.append(line_no_comment)
    return '\n'.join(lines)


def process_file(relative_path: str, project_root: str, output_dir: str, processed_files: set):
    """
    Recursively processes a file:
    1. Checks if it has already been processed.
    2. Determines its absolute source and destination paths.
    3. If a .tex file, cleans it, writes it, and finds dependencies.
    4. If a .bib file, cleans it and writes it.
    5. If any other file (image, .cls, .sty), copies it directly.
    6. Recursively calls itself for all found dependencies.
    """
    if relative_path in processed_files:
        return  # Avoid redundant processing or circular dependencies
    
    abs_src_path = os.path.normpath(os.path.join(project_root, relative_path))
    
    # Handle files included without extensions (e.g., \input{sections/intro})
    if not os.path.exists(abs_src_path):
        if not relative_path.endswith('.tex') and os.path.exists(abs_src_path + '.tex'):
            relative_path += '.tex'
            abs_src_path += '.tex'
        elif not relative_path.endswith('.bib') and os.path.exists(abs_src_path + '.bib'):
            relative_path += '.bib'
            abs_src_path += '.bib'
        elif not os.path.exists(abs_src_path):
             # Could be an image file where graphicx finds the extension
             # We just log it and assume it's fine, but we won't find it to copy
             # A more robust script would check for .png, .pdf, .jpg, etc.
             # For this script, we assume the *exact* path is given or it's a .tex/.bib
             logging.warning(f"File not found and extension-less: '{relative_path}'. Skipping.")
             return

    # Add to processed set *after* extension resolution
    processed_files.add(relative_path)
    
    abs_dest_path = os.path.normpath(os.path.join(output_dir, relative_path))
    
    # Create the destination directory if it doesn't exist
    try:
        os.makedirs(os.path.dirname(abs_dest_path), exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create directory {os.path.dirname(abs_dest_path)}: {e}")
        return

    file_ext = os.path.splitext(relative_path)[1].lower()
    dependencies = set()

    try:
        # For text files, read, clean, parse, and write
        if file_ext in ['.tex', '.bib']:
            logging.info(f"Cleaning {relative_path} -> {abs_dest_path}")
            with open(abs_src_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            cleaned_content = clean_tex_content(content)
            
            with open(abs_dest_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            
            # Only .tex files have dependencies we need to parse
            if file_ext == '.tex':
                # IMPORTANT: We parse the *original* content, not the cleaned one,
                # as comments might contain valid temporary \input commands.
                dependencies = find_dependencies(strip_comments_for_dependency_parsing(content), project_root)
        
        # For all other files, just copy them
        elif file_ext not in ['.aux', '.log', '.out', '.bbl', '.blg', '.toc', '.lof', '.lot', '.synctex.gz']:
             logging.info(f"Copying {relative_path} -> {abs_dest_path}")
             shutil.copy2(abs_src_path, abs_dest_path)

    except FileNotFoundError:
        logging.error(f"File not found during processing: {abs_src_path}")
    except Exception as e:
        logging.error(f"Error processing file {relative_path}: {e}")

    # Recurse for all found dependencies
    for dep in dependencies:
        # We assume all paths are relative to the project_root
        process_file(dep, project_root, output_dir, processed_files)


def main():
    """
    Main function to parse arguments and start the cleaning process.
    """
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    parser = argparse.ArgumentParser(description="Clean a LaTeX project for submission.")
    parser.add_argument("main_tex_file", 
                        help="The main .tex file of the project (e.g., main.tex)")
    parser.add_argument("output_dir", 
                        help="The directory to save the cleaned project (e.g., latex_file_clean)")
    args = parser.parse_args()

    # Get absolute paths
    main_file_path = os.path.abspath(args.main_tex_file)
    project_root = os.path.dirname(main_file_path)
    main_file_relative = os.path.basename(main_file_path)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(main_file_path):
        logging.critical(f"Main file not found: {main_file_path}")
        return

    # Start with a clean slate
    if os.path.exists(output_dir):
        logging.info(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    os.makedirs(output_dir)

    processed_files = set()
    
    logging.info(f"--- Starting LaTeX Project Cleanup ---")
    logging.info(f"Project Root: {project_root}")
    logging.info(f"Output Dir:   {output_dir}")
    
    # Start the recursive process
    process_file(main_file_relative, project_root, output_dir, processed_files)
    
    logging.info(f"--- Cleanup Complete ---")
    logging.info(f"Processed {len(processed_files)} files.")
    logging.info(f"Cleaned project saved to: {output_dir}")


if __name__ == "__main__":
    main()
