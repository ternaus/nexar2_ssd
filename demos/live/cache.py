""" cache.py

sets up the demo with the server ip address

Ellis Brown, Max deGroot
"""


import argparse
import ipaddress
import os.path
import sys

DIR = os.path.dirname(os.path.realpath(__file__))
IP_FILE = DIR + '/ip_cache.txt'


def parse_args(argv=None):
    """parses arg values"""
    parser = argparse.ArgumentParser(
        description="""Configure demo example with Server IP address.
        (note: the previous ip will be reloaded if none is passed)""")
    parser.add_argument('ip', nargs='?',
                        help='(optional) ip address of server machine')
    return parser.parse_args(argv)


def load_ip():
    """loads the previous ip, assuming no ip passed in"""
    assert os.path.isfile(IP_FILE), IP_FILE + " not found and no ip supplied."
    with open(IP_FILE, "r") as f:
        ip = f.read()
    return ipaddress.ip_address(ip.strip())


def cache_ip(ip):
    """stores the server ip addr in the IP_FILE"""
    with open(IP_FILE, "w") as f:
        f.write(ip)


def server_ip(address=None):
    """returns the server ip for use with live demo. caches ip's in IP_FILE.

    Args:
        address (optional, string): ip address of server machine
            if supplied: cache it in IP_FILE
            else: load ip from IP_FILE
    Return:
        ipaddress (ipaddress.IPv4Address OR ipaddress.IPv6Address)
            type of ipaddress will depend on format of ip supplied
    """
    if address:
        # ip address supplied, store it for next run
        cache_ip(address)
        ip = ipaddress.ip_address(address)
    else:
        ip = load_ip()
    print('Using server ip:', ip)
    return ip


if __name__ == "__main__":
    args = parse_args()
    server_ip(args.ip)
