import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="bapsap",
    version="0.0.1",
    author="Livin Nector",
    author_email="livinnector@gmail.com",
    description=("An app to solve and play ballsort puzzle game."),
    license="MIT",
    keywords="gameplaying puzzle search",
    # url = "project_url",
    package_dir={"":"src"}, 
    packages=["bapsap"],
    install_requires=[
        "numpy",
        "scikit-image",
        "scikit-learn",
        "pure-python-adb"
    ],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Planning" "Topic :: Utilities",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",   
    ],
)
