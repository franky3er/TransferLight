import os
import subprocess
from typing import List

from src.params import PYTHON, RANDOM_TRIPS_SCRIPT


class RandomTrips:

    def __call__(self, *args):
        args = [str(arg) for arg in args]
        self._random_trips(args)

    @staticmethod
    def _random_trips(args: List[str]):
        cmd = [PYTHON, RANDOM_TRIPS_SCRIPT] + args
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


randomTrips = RandomTrips()
