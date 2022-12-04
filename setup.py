from setuptools import setup
from setuptools import find_packages
#TODO: Add ls and esm and gvpdir
setup(
    name="prot_split",
    packages=[
        'utils',
        # 'esmdir',
    ],
    package_dir={
        'utils': './utils',
        # 'esmdir': './esmdir',
    },
)
