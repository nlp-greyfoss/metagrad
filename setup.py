import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='metagrad',
      version='0.0.1',
      description='一个用于学习的仿Pytorch纯Python实现的自动求导工具。',
      author='nlp-greyfoss',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages=['core'],
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License"
      ],
      install_requires=['numpy'],
      python_requires='>=3.8',
      extras_require={
          'testing': [
              "pytest",
              "torch"
          ],
      },
      include_package_data=True)
