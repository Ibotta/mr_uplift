"""
Run setup
"""

from setuptools import setup, find_packages

def parse_description(description):
    """
    Strip figures and alt text from description
    """
    return "\n".join(
        [
        a for a in description.split("\n")
        if ("figure::" not in a) and (":alt:" not in a)
        ])

DISTNAME = "mr_uplift"
VERSION = "0.0.10"
DESCRIPTION = "Machine learning tools for uplift models"
with open("README.rst") as f:
    LONG_DESCRIPTION = parse_description(f.read())
CLASSIFIERS = [
    "Intended Audience :: Developers",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering"
    ]
AUTHOR = "Ibotta Inc."
AUTHOR_EMAIL = "machine_learning@ibotta.com"
LICENSE = "Apache 2.0"
DOWNLOAD_URL = "https://github.com/Ibotta/mr_uplift"
PROJECT_URLS = {
    "Source Code": "https://github.com/Ibotta/mr_uplift"
    }
MIN_PYTHON_VERSION = "3.5"

setup(name=DISTNAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      classifiers=CLASSIFIERS,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      license=LICENSE,
      download_url=DOWNLOAD_URL,
      project_urls=PROJECT_URLS,
      packages=find_packages(),
      install_requires=[
          'numpy>1.17.2',
          'pandas>=0.20.0',
          'tensorflow>=2.2.0',
          'keras',
          'dill',
          'sklearn'
      ],
      python_requires=">={0}".format(MIN_PYTHON_VERSION)
      )
