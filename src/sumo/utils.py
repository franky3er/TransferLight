import subprocess

from src.params import PYTHON, RANDOM_TRIPS_SCRIPT


def netgenerate(*args):
    cmd = ["netgenerate"] + [str(arg) for arg in args]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def netconvert(*args):
    cmd = ["netconvert"] + [str(arg) for arg in args]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def randomTrips(*args):
    cmd = [PYTHON, RANDOM_TRIPS_SCRIPT] + [str(arg) for arg in args]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
