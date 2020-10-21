"""Script for packaging the project."""

from pathlib import Path

from setuptools import setup, find_packages

PACKAGE_NAME = "tests_wo_split"


def get_version():
    """"Convenient function to get the version of this package."""

    ns = {}
    version_path = Path(PACKAGE_NAME) / 'version.py'
    if not version_path.is_file():
        return
    with open(version_path) as version_file:
        exec(version_file.read(), ns)
    return ns['__version__']


dependencies = (
    "cvxopt",
    "matplotlib",
    "numpy",
    "pyyaml",
    "scipy",
    "sklearn",
    "torch"
)
setup(
    name=PACKAGE_NAME,
    version=get_version(),
    packages=find_packages(),
    description="Learning kernel tests without data splitting.",
    url="https://github.com/MPI-IS/tests-wo-splitting",
    author="Jonas Kübler",
    author_email="jonas.m.kuebler@tuebingen.mpg.de",
    python_requires=">=3.5",
    install_requires=dependencies,
    long_description=open('README.md').read(),
    license="MIT License",
)
