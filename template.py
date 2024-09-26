import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    "setup.py",
    "app.py",
    "store_index.py",
    "static/style.css",
    "templates/chat.html"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir:
        if not os.path.exists(filedir):
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Created directory: {filedir} for the file {filename}")
        else:
            logging.info(f"Directory already exists: {filedir}")

    if not filepath.exists() or filepath.stat().st_size == 0:
        with open(filepath, 'w') as f:
            pass
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists and is not empty: {filename}")