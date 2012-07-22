#!/home/amitibo/epd/bin/python

from setuptools import setup


def main():
    """main setup function"""
    
    setup(
        name='compsense',
        version='0.1',
        description='A toolbox for testing sparse reconstruction algorithms',
        author='Amit Aides',
        author_email='amitibo@tx.technion.ac.il',
        packages=['compsense'],
        package_data={'compsense': ['data/*.*']},
        )


if __name__ == '__main__':
    main()
