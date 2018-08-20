from os import path

from setuptools import setup, find_packages

import survive


def load_file(filename):
    filename = path.join(path.abspath(path.dirname(__file__)), filename)
    with open(filename, "r") as file:
        return file.read()


setup(
    name="survive",
    version=survive.__version__,
    description="Survival analysis in Python",
    long_description=load_file("README.rst"),
    long_description_content_type="text/x-rst",
    url="https://github.com/artemmavrin/survive",
    author="Artem Mavrin",
    author_email="amavrin@ucsd.edu",
    packages=sorted(find_packages()),
    include_package_data=True,
    install_requires=["numpy", "scipy", "pandas", "matplotlib"],
    extras_require={"seaborn": ["seaborn"]},
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)
