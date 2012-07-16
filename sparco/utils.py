"""

"""

from __future__ import division
import pkg_resources

__all__ = [
    'getResourcePath'
    ]


def getResourcePath(name):
    """return the path to resource"""

    return pkg_resources.resource_filename(__name__, "data/%s" % name)
    

def main():
    """
    Main Function
    """

    pass


if __name__ == '__main__':
    main()