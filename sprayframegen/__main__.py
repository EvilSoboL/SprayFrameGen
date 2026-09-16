"""python -m sprayframegen: настройки, проверка и экспорт --generate FILE."""

import argparse
from dataclasses import asdict
import json
import sys
import time

from . import __version__
from .configuration import ConfigurationError, load, new_configuration, save
from .ui import InterfaceError
from .generation.controller import ProgressEvent, run_generation


def _progress_printer(event: ProgressEvent) -> None:
    if event.total_frames > 0:
        pct = event.completed_frames * 100 // event.total_frames
    else:
        pct = 0
    elapsed = f"{event.elapsed_seconds:.1f}с"
    print(f"\rКадр {event.frame_number}/{event.total_frames} ({pct}%) — готово {event.completed_frames}, {elapsed}", end="", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SprayFrameGen — генератор кадров распыла")
    parser.add_argument("--version", action="version", version=__version__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check", metavar="FILE", help="проверить конфигурацию")
    actions.add_argument("--new", metavar="FILE", help="сохранить новую конфигурацию")
    actions.add_argument("--generate", metavar="FILE", help="сформировать серию по JSON с выбранными эффектами")
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
        elif args.generate:
            result = load(args.generate)
            for warning in result.warnings:
                print("Предупреждение: " + warning)
            print("Генерация серии...")
            directory = run_generation(result.configuration, progress_callback=_progress_printer)
            print()  # новая строка после прогресс-бара
            print(f"Серия сохранена: {directory}")
        elif args.smoke_test or args.environment:
            from .environment import current_environment
            from .generation import prepare
            prepare(new_configuration(seed=0))
            print(json.dumps(asdict(current_environment()), ensure_ascii=False, indent=2))
        else:
            from .ui.app import run
            run()
    except (ConfigurationError, InterfaceError, OSError, UnicodeError, ImportError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
