"""UMI maintenance and troubleshooting commands."""

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog='umi-maint', description='场景与标定维护工具',
                                     epilog='使用 umi-maint {scene|calib} --help 查看各组命令。')
    parser.add_argument('area', choices=('scene', 'calib'), help='选择场景或标定维护命令')
    area = parser.parse_args(arguments[:1]).area
    if area == 'scene':
        from umi.apps import scene
        return scene.main(arguments[1:], maintenance=True)
    from umi.apps import calib
    return calib.main(arguments[1:], prepare_environment=True, maintenance=True)


if __name__ == '__main__':
    raise SystemExit(main())
