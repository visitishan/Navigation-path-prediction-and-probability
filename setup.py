from setuptools import setup, find_packages

setup(
    packages=find_packages(include=["markovpath", "markovpath.*"]),
    py_modules=["path_nav"],
)
