#!/usr/bin/env sh
# CPU Tests. Temporary change dir.
(cd cpu && python3 Test.py)
(cd quantizer && python3 genTest.py)