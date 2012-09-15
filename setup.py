# -*- coding: utf-8 -*-
"""
pycompsense: A toolbox for compressed sensing and sparse reconstruction algorithms.

Copyright (C) 2012 Amit Aides
Author: Amit Aides <amitibo@tx.technion.ac.il>
URL: <http://bitbucket.org/amitibo/pycompsense>
License: See attached license file
"""

from setuptools import setup


PACKAGE_NAME = 'compsense'
VERSION = '0.1'
DESCRIPTION = 'A toolbox for compressed sensing and sparse reconstruction algorithms'
AUTHOR = 'Amit Aides'
EMAIL = 'amitibo@tx.technion.ac.il'
URL = "http://bitbucket.org/amitibo/pycompsense"

def main():
    """main setup function"""
    
    setup(
        name=PACKAGE_NAME,
        version=VERSION,
        description=DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        packages=[PACKAGE_NAME],
        package_data={PACKAGE_NAME: ['data/*.*']},
        )


if __name__ == '__main__':
    main()
