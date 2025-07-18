import logging
import os
import sys
import traceback
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import Optional

import typer

from .midi_worker import generate, run, load_model, finish_run


app_version = "0.1.0"


class MIDIWorkerException(Exception):
    """Generic MyApp exception"""

    pass


worker_cli = typer.Typer(
    help="This is a cli for remote work",
    invoke_without_command=True,
    no_args_is_help=True,
)


def main():
    load_model()
    midi_seq, seed = run()
    finish_run()
