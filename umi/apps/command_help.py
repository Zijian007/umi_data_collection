"""Help conventions for daily commands and the maintenance entry point."""

from argparse import SUPPRESS


def command(commands, name, summary, *, visible=True):
    help_option = {'help': summary} if visible else {}
    return commands.add_parser(name, description=summary, **help_option)


def expert_help(maintenance, message):
    return message if maintenance else SUPPRESS
