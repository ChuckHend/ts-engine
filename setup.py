#!/usr/bin/env python

from distutils.core import setup

setup(name='forecast-engine',
      version='0.1dev',
      long_description=open('README.md').read(),
      author='Adam Hendel',
      author_email='hendel.adam@gmail.com',
      url='https://github.com/ChuckHend/forecast-engine',
      packages=['src'],
      install_requires=[
      	'os', 
      	'time', 
      	'datetime', 
      	'numpy', 
      	'pandas', 
      	'pandas-datareader', 
      	'matplotlib', 
      	'scikit-learn', 
      	'scipy', 
      	'bs4', 
      	'keras', 
      	'tensorflow-gpu'
      ],
      license='GNU General Public License v3.0'
     )
