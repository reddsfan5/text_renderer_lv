"""Command-line entry point for text dataset generation."""

import argparse
import sys


def _run_single_line(args):
    from text_renderer.pipelines.single_line import main

    return main(args)


def _run_callnumber_multiline(args):
    from text_renderer.pipelines.callnumber_multiline_lmdb import main

    return main(args)


def _run_callnumber_detection(args):
    from text_renderer.pipelines.callnumber_detection import main

    return main(args)


COMMANDS = {
    "single-line": (_run_single_line, "generate configurable single-line OCR data"),
    "callnumber-multiline": (
        _run_callnumber_multiline,
        "generate multi-line call-number recognition LMDB data",
    ),
    "callnumber-detection": (
        _run_callnumber_detection,
        "generate call-number detection data in LabelMe format",
    ),
}


def build_parser():
    parser = argparse.ArgumentParser(prog="text-renderer", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_, help_text) in COMMANDS.items():
        subparsers.add_parser(name, add_help=False, help=help_text)
    return parser


def main(arg_list=None):
    argv = list(sys.argv[1:] if arg_list is None else arg_list)
    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0

    args = parser.parse_args(argv[:1])
    handler, _ = COMMANDS[args.command]
    return handler(argv[1:])


if __name__ == "__main__":
    main()
