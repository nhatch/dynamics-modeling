from distutils.core import setup
import os
from setuptools import find_packages

# This is necessary to lookup pip with pip -e file. It also makes package_dir unnecessary
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "package"))

setup(
    name='rosbag2torch',
    version='0.0.1',
    description='Conversion of rosbags into sequences that can be made into pytorch datasets',
    author='Jakub Filipek',
    author_email='balbok@cs.washington.edu',
    # packages=['rosbag2torch', 'rosbag2torch.*'],
    packages=find_packages("."),
    # package_dir={'rosbag2torch': 'src/package/rosbag2torch'},
    install_requires=[
        "numpy",
        "rospy-all",
        "rosbag",
        "pycryptodomex",
        "python-gnupg",
        "h5py",
        "tqdm",
        "scipy",
        "torch",
        "roslz4",
    ],
)
