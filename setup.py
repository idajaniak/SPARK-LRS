#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from setuptools import setup, find_packages

setup(
    name='spark_lrs',        
    version='0.1.0',             
    author='Ida A. Janiak',
    author_email='ida.janiak@manchester.ac.uk',
    description='A pipeline to perform automated data reduction of files from TNT/LRS.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'astropy>=2.0',
        'matplotlib',
        'ccdproc',
        'scipy',
        'scikit-image',
        'astroscrappy',
        'reproject',
        'scikit-learn',
        'pandas',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

