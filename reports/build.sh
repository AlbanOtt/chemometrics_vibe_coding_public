#!/bin/bash
# Build script for integrated book and presentation

set -e

echo "Building main book..."
quarto render --no-execute

echo "Building presentation..."
cd presentation
quarto render
cd ..

echo "Build complete! Output in _book/"
echo "  - Book: _book/index.html"
echo "  - Presentation: _book/presentation/workshop_presentation.html"
