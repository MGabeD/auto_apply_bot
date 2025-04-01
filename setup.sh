#!/bin/bash/env bash

set -e

echo "Installing dependencies... "
pip install -e .[test]

echo "Installing pre-commit hooks..."
pre-commit install

echo "Setup complete!"

