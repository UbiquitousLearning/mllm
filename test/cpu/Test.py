import argparse
import importlib
import os

from TestUtils import TestBase

for file in os.listdir("./"):
    if file.endswith("Test.py") and file != "__init__.py":
        importlib.import_module(file[:-3])
cwd_basepath = os.path.basename(os.getcwd())
if cwd_basepath != "bin":
    os.chdir(os.path.realpath("../../bin"))
parser = argparse.ArgumentParser(description='Run tests')
parser.add_argument('--file', type=str, help='Run test defined in [file], partial match.', nargs='+')
if __name__ == '__main__':
    args = parser.parse_args()
    print(TestBase.__subclasses__())
    cls = TestBase.__subclasses__()
    tests = []
    if args.__contains__("name"):
        for c in cls:
            for name in args.name:
                if name in c.__name__:
                    tests.append(c())
    else:
        tests = [instance() for instance in cls]

    for test in tests:
        test.test()
