from setuptools import setup
import sys

setup(
    name='rl_rvo_nav',
    py_modules=['rl_rvo_nav'],
    version= '2.0',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'transforms3d',
        'gym==0.20.0',
        'Pathlib',
        'mpi4py',
    ],
    description="source code of the paper",
    author="Han Ruihua",
)