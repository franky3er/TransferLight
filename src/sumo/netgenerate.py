import os
import pathlib
import subprocess
from typing import Union, List, Tuple

import sumolib

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.uniform import Uniform


np.random.seed(42)


class NetGenerate:

    def __call__(self, *args):
        args = [str(arg) for arg in args]
        if args[0] in ["-g", "--grid", "-s", "--s", "-r", "--rand"]:
            self._netgenerate(args)
        elif args[0] in ["-i", "--isolated"]:
            self._netgenerate_isolated_intersection(args)

    @staticmethod
    def _netgenerate(args: List[str]):
        cmd = ["netgenerate"] + args
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @staticmethod
    def _netconvert(args: List[str]):
        cmd = ["netconvert"] + args
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _netgenerate_isolated_intersection(self, args):
        args[0] = "-r"
        self._netgenerate(args)
        net_xml_path = get_arg_value(args, ["-o", "--output-file"])
        net = sumolib.net.readNet(net_xml_path)
        intersections = [intersection for intersection in net.getNodes()
                         if len(intersection.getIncoming()) >= 3 and len(intersection.getOutgoing()) >= 3]
        rand_intersection = np.random.choice(intersections)
        rand_intersection_approaches = rand_intersection.getIncoming() + rand_intersection.getOutgoing()
        rand_intersection_approaches = [approach.getID() for approach in rand_intersection_approaches]
        args_netconvert = ["-s", net_xml_path, "-o", net_xml_path]
        args_netconvert += ["--tls.discard-simple", "--keep-edges.explicit", ",".join(rand_intersection_approaches)]
        self._netconvert(args_netconvert)


def get_arg_value(args: List[str], key: Union[str, List[str]]):
    if isinstance(key, str):
        key = [key]
    for k in key:
        if k in args:
            k_idx = args.index(k)
            val = args[k_idx+1] if k_idx < len(args)-1 else None
            if val.startswith("-"):
                return True
            return val
    return None

def arg_exists(args: List[str], key: Union[str, List[str]]):
    if isinstance(key, str):
        key = [key]
    for k in key:
        if k in args:
            return True
    return False


netgenerate = NetGenerate()
