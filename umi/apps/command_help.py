"""Group command summaries by purpose on the top-level help page."""

from argparse import ArgumentParser


def grouped_help(commands: dict[str, ArgumentParser], daily: tuple[str, ...]) -> str:
    width = max(map(len, commands))
    maintenance = [name for name in commands if name not in daily]
    groups = []
    for title, names in (('日常操作', daily), ('维护与诊断', maintenance)):
        lines = [title]
        for name in names:
            summary = commands[name].description.splitlines()[0]
            lines.append(f'  {name:<{width}}  {summary}')
        groups.append('\n'.join(lines))
    return '\n\n'.join(groups)
