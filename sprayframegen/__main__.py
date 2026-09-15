"""python -m sprayframegen [--check FILE | --new FILE | --smoke-test]."""

import argparse
from dataclasses import asdict
import json
import sys

from . import __version__
from .configuration import ConfigurationError, load, new_configuration, save


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SprayFrameGen — настройки генератора распыла")
    parser.add_argument("--version", action="version", version=__version__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", metavar="FILE", help="проверить конфигурацию")
    actions.add_argument("--new", metavar="FILE", help="сохранить новую конфигурацию")
    actions.add_argument("--smoke-test", action="store_true", help="проверить каркас без открытия окна")
    actions.add_argument("--environment", action="store_true", help="показать окружение и сборку")
    args = parser.parse_args(argv)
    try:
        if args.check:
            result = load(args.check)
            print("Конфигурация корректна.")
            for warning in result.warnings:
                print("Предупреждение: " + warning)
        elif args.new:
            save(new_configuration(), args.new)
            print(f"Настройки сохранены: {args.new}")
        elif args.smoke_test or args.environment:
            from .environment import current_environment
            from .generation import prepare
            prepare(new_configuration(seed=0))
            print(json.dumps(asdict(current_environment()), ensure_ascii=False, indent=2))
        else:
            from .ui.app import run
            run()
    except (ConfigurationError, OSError, UnicodeError, ImportError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
