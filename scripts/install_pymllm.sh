pip wheel -v -w dist .
pip install dist/*.whl --force-reinstall
