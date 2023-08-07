import gzip
from sumolib import net
from xml.sax import parse


class NetState(net.Net):

    def __init__(self):
        super(NetState, self).__init__()

    def init(self):
        print("hello")


class NetStateReader(net.NetReader):

    def __init__(self, **others):
        super(NetStateReader, self).__init__(**others)
        self._net = others.get("net", NetState())


def readNetState(filename, **others) -> NetState:
    netreader = NetStateReader(**others)
    try:
        parse(gzip.open(filename), netreader)
    except IOError:
        parse(filename, netreader)
    net = netreader.getNet()
    net.init()
    return net
