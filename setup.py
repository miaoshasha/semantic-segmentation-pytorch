from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['torchvision']
DEPENDENCY_LINKS = ['http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    packages=find_packages(),
    include_package_data=True,
    description='My keras trainer application package.'
)
