from setuptools import find_packages, setup

setup(
    name="pytolerance",
    version="0.1",
    description="A Python library for calculating tolerance intervals, and other useful statistics.",
    author="Jorgen Wu",
    url="https://github.com/wujorgen/pytolerance",
    packages=find_packages(),
    entry_points={
       'console_scripts': [
           'DataDrift = DataDrift.__init__:main',
       ],
    }
)