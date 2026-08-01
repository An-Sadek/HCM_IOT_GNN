"""Dedicated CLI entry point for training the traffic-adapted HTGNN."""

import run_train


if __name__ == "__main__":
    run_train.main(default_architecture="htgnn")
