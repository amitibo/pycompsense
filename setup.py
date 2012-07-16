#!/home/amitibo/epd/bin/python

from setuptools import setup


def main():
    """main setup function"""
    
    setup(
        name='sparco',
        version='0.1',
        description='A toolbox for testing sparse reconstruction algorithms',
        author='Amit Aides',
        author_email='amitibo@tx.technion.ac.il',
        packages=['sparco'],
        package_data={'sparco': ['data/*.*']},
        )


if __name__ == '__main__':
    main()
