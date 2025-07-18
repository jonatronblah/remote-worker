import logging
import os
import sys
import traceback
from enum import Enum
from pathlib import Path
from pprint import pprint
from typing import Optional

import typer

from remote_worker.midi_model.midi_worker import run, finish_run, load_model, get_model_path 

app_version = "0.1.0"


class MIDIWorkerException(Exception):
    """Generic MyApp exception"""

    pass

class MIDIWorker:
    version = app_version
    name = "Midi-Worker"

    def __init__(self, midi_filepath) -> None:
        pass

    def load_midi(self):
        


worker_cli = typer.Typer(
    help="This is a cli for remote work",
    invoke_without_command=True,
    no_args_is_help=True,
)


def main():
    load_model()
    midi_seq, seed = run()
    finish_run()
