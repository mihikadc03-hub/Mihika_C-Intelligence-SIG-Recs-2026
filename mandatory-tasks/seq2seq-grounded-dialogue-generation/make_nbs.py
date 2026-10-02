import json
import os

def create_nb(filename, code_content, title):
    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", "This notebook contains the implementation."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in code_content.split("\n")]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filename, 'w') as f:
        json.dump(nb, f, indent=2)

subtasks = [
    ("subtask1", "Subtask 1: Basic Seq2Seq"),
    ("subtask2", "Subtask 2: Attention Seq2Seq"),
    ("subtask3", "Subtask 3: Grounded Dialogue Seq2Seq")
]

for folder, title in subtasks:
    model_path = os.path.join(folder, "model.py")
    if os.path.exists(model_path):
        with open(model_path, 'r') as f:
            code = f.read()
        nb_path = os.path.join(folder, f"{folder}_notebook.ipynb")
        create_nb(nb_path, code, title)
        print(f"Created {nb_path}")
