import json
import argparse
import os

def clean_notebook_images(file_path):
    """
    Removes base64 image data from 'image/png' outputs in a Jupyter Notebook.

    Args:
        file_path (str): The path to the .ipynb file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Is it a valid notebook file?")
        return
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    cleaned_cells = 0
    for cell in notebook.get('cells', []):
        if 'outputs' in cell:
            for output in cell.get('outputs', []):
                if 'data' in output and 'image/png' in output['data']:
                    output['data']['image/png'] = ""
                    cleaned_cells += 1

    if cleaned_cells > 0:
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
                # ipynb files should have a newline at the end
                f.write('\n')
            print(f"Cleaned {cleaned_cells} image outputs from '{os.path.basename(file_path)}'.")
        except IOError as e:
            print(f"Error writing to file {file_path}: {e}")
    else:
        print(f"No 'image/png' outputs to clean in '{os.path.basename(file_path)}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove base64-encoded image/png data from Jupyter Notebook files to reduce size."
    )
    parser.add_argument(
        "files",
        nargs='+',
        help="One or more notebook files to clean (e.g., your_notebook.ipynb)."
    )
    args = parser.parse_args()

    for file in args.files:
        if os.path.exists(file):
            clean_notebook_images(file)
        else:
            print(f"File not found: {file}")
