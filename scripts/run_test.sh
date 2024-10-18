#!/bin/bash
for file in ../bin/*Test ../bin/*TEST; do
    if [ -x "$file" ]; then
    echo "Running $file..."
    "../bin/$file"
    else
    echo "Skipping non-executable $file..."
    fi
done