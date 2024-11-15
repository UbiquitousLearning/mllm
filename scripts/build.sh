#!/bin/bash
mkdir ../build
cd ../build || exit

cmake .. -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
