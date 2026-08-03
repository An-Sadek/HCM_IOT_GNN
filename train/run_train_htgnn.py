"""Train HTGNN with separated dynamic and positional features.

Usage from the repository root::

    python train/run_train_htgnn.py --epochs 100 --horizon 12
"""

import run_train


if __name__ == "__main__":
    run_train.main(default_architecture="htgnn")
