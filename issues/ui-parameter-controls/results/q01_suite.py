"""Воспроизводимый общий прогон с согласованным исключением нагрузки.

Из корня: .venv/Scripts/python.exe issues/ui-parameter-controls/results/q01_suite.py
Пользователь разрешил исключение 26.10.2026 при приёмке ISSUE-001.
"""

import sys
import argparse
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def cases(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from cases(test)
        else:
            yield test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--without-d09", action="store_true", help="Запустить независимо все модули кроме test_acceptance")
    args = parser.parse_args()
    selected = []
    for test in cases(unittest.defaultTestLoader.discover("tests")):
        if args.without_d09 and test.id().startswith("test_acceptance."):
            continue
        if test.id().endswith(".test_load_1000x10000_memory_time"):
            print("DEFERRED", test.id(), flush=True)
        else:
            selected.append(test)
    print("RUNNING", len(selected), "tests", flush=True)
    filename = "q01-core-suite.log" if args.without_d09 else "q01-suite.log"
    with Path("issues/ui-parameter-controls/results", filename).open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.TestSuite(selected))
    print("RESULT", result, flush=True)
    raise SystemExit(not result.wasSuccessful())
