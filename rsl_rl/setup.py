from setuptools import find_packages
from distutils.core import setup

setup(name='rsl_rl',
      version='2.0.0',
      author='Nikita Rudin',
      author_email='rudinn@ethz.ch',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='Fast and simple RL algorithms implemented in pytorch',
      python_requires='>=3.6',
      install_requires=[
            "torch>=1.10.0",
            "torchvision>=0.5.0",
            "numpy>=1.16.4",
            "GitPython",
            "onnx"],
)
