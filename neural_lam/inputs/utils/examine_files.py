# Standard library
from pathlib import Path

# Third-party
import numpy as np
import torch


def tree(root_dir, prefix="", fn=lambda fp: fp.name):
    """
    Generates a tree structure for the given directory using pathlib.Path.

    Parameters:
    - root_dir: The root directory to generate the tree from, as a Path object or a string.
    - prefix: The prefix to use for the current level of the tree.
    """
    root_dir = Path(root_dir)  # Ensure root_dir is a Path object
    if prefix == "":  # Indicates first call, root of the tree
        print(root_dir)

    entries = list(sorted(root_dir.iterdir(), key=lambda x: x.name))
    # Filter out hidden files and directories
    entries = [e for e in entries if not e.name.startswith(".")]
    for i, entry in enumerate(entries):
        connector = "├──" if i < len(entries) - 1 else "└──"
        print(f"{prefix}{connector} {fn(entry)}")
        if entry.is_dir():  # If entry is a directory, recurse
            new_prefix = prefix + ("│   " if i < len(entries) - 1 else "    ")
            tree(entry, new_prefix, fn=fn)


def main(fp_root):
    def return_filename_and_shape(fp):
        fp = Path(fp)
        shape = None

        def get_shapes(data):
            if isinstance(data, dict):
                return {k: get_shapes(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [get_shapes(v) for v in data]
            elif isinstance(data, torch.Tensor):
                return list(data.shape)
            elif isinstance(data, np.ndarray):
                return list(data.shape)
            else:
                return str(data)

        if fp.name.endswith(".pt"):
            d = torch.load(fp)
            shape = get_shapes(d)
        elif fp.name.endswith(".npy"):
            shape = get_shapes(np.load(fp))

        if shape is not None:
            return f"{str(fp.name)}  {shape}"
        else:
            return str(fp.name)

    tree(fp_root, fn=return_filename_and_shape)


if __name__ == "__main__":
    # Standard library
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="/nwp/neural-lam")
    args = parser.parse_args()
    main(fp_root=args.root_dir)
