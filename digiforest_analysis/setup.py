# from distutils.core import setup
# from catkin_pkg.python_setup import generate_distutils_setup

# d = generate_distutils_setup(packages=["digiforest_analysis"], package_dir={"": "src"})

# setup(**d)

# from setuptools import find_packages
from distutils.core import setup

setup(
    name="digiforest_analysis",
    version="0.0.0",
    author="Matias Mattamala",
    author_email="matias@robots.ox.ac.uk",
    packages=["digiforest_analysis"],
    package_dir={"": "src"},
    python_requires=">=3.6",
    description="Python tools for DigiForest",
)
