from setuptools import setup, find_packages

try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

#import uuid

REQUIRED_PACKAGES = ['torchvision']
DEPENDENCY_LINKS = ['http://download.pytorch.org/whl/cpu/torch-0.3.0.post4-cp27-cp27mu-linux_x86_64.whl']

setup(
    description='segmentation network using pytorch',
    author='shasha',
    url='',
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    packages=find_packages(),
    include_package_data=True,
)
