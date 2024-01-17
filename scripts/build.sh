#!/bin/bash
mkdir ../build
cd ../build || exit

cmake .. -DCMAKE_BUILD_TYPE=Release -DTEST=OFF

make -j4
