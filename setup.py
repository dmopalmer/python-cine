try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
        name='python-cine',
        version='0.1',
        packages=['cine'],
        url='',
        license='GPL3',
        author='cbenhagen; palmer',
        author_email='palmer@lanl.gov',
        description='Read Phantom cine files with a PIMS interface',
        install_requires=[
            'docopt',
            'numpy',
            'pims',
            'pytz']
)
