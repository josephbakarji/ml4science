import glob
import os
import re

def process_latex_in_file(file_path):
    # Read the file
    with open(file_path, 'r') as file:
        content = file.read()

    # Replace single dollar signs with double dollar signs for LaTeX
    # This regex finds non-greedy matches between single dollar signs
    # and replaces them with double dollar signs
    modified_content = re.sub(r'\$(.*?)\$', r'$$\1$$', content)

    return modified_content

def main():
    # Get current working directory
    root_dir = '/Users/josephbakarji/Documents/academic/classes/teaching/ml4science/spr-24/website/ml4science/'
    hw_dir = os.path.join(root_dir, 'content', 'hw')
    hw_local_dir = os.path.join(root_dir, 'content', 'hw_local')

    # Find all markdown files with '_local.md' in the current directory
    files = glob.glob(os.path.join(hw_local_dir, '*_local.md'))
    print(files)

    for file_path in files:
        modified_content = process_latex_in_file(file_path)

        # Create new file name by removing '_local' from the original file name
        new_file_path = file_path.replace('_local', '')
        print(file_path)
        print(new_file_path)

        # Write the modified content to the new file
        with open(new_file_path, 'w') as new_file:
            new_file.write(modified_content)
        print(f"Processed {file_path} -> {new_file_path}")

if __name__ == "__main__":
    main()
