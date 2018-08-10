from os import path

from setuptools import setup, find_packages

import survive

# Load long description from README.md
here = path.abspath(path.dirname(__file__))
with open(path.join(here, "README.md"), "r") as f:
    long_description = f.read()

setup(
    name="survive",
    version=survive.__version__,
    description="Survival analysis in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/artemmavrin/survive",
    author="Artem Mavrin",
    author_email="amavrin@ucsd.edu",
    packages=sorted(find_packages(exclude=("*.test",))),
    include_package_data=True,
    install_requires=["numpy", "scipy", "pandas", "matplotlib"],
    extras_require={"seaborn": ["seaborn"]}
)
