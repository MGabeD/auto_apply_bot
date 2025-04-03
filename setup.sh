#!/usr/bin/env bash

set -e

echo "Installing dependencies..."

OS_TYPE="$(uname -s)"

if [[ "$OS_TYPE" =~ MINGW|MSYS|CYGWIN ]]; then
    echo "Detected Windows (Git Bash or similar)"
    echo "Installing PyTorch with CUDA 12.1 support..."
    python -m pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -e .[dev,windows]

elif [[ "$OS_TYPE" == "Linux" ]]; then
    echo "Detected Linux"
    echo "Installing PyTorch with CUDA 12.1 support..."
    python -m pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -e .[dev,unix]

elif [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "Detected macOS"
    echo "Installing CPU-only PyTorch (CUDA not supported on macOS)..."
    python -m pip install torch==2.1.2
    python -m pip install -e .[dev,unix]

else
    echo "Unknown OS type: $OS_TYPE"
    echo "Falling back to installing 'dev' extras only"
    python -m pip install -e .[dev]
fi

echo "Installing pre-commit hooks..."
pre-commit install

echo "Setup complete!"
