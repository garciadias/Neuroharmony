#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from os import system
with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements:
    requirements = requirements.readlines()

test_requirements = [
    "pytest",
]

setup(
    name='Neuroharmony',
    version='0.0.0',
    description="A tool to perform Freesurfer volume Harminization.",
    long_description=readme,
    author="Rafael Garcia-Dias",
    author_email='rafaelagd@gmail.com',
    url='https://github.com/garciadias/Neuroharmony',
    packages=[
        'neuroharmony', 'neuroharmony.models', 'neuroharmony.data'
    ],
    package_dir={'neuroharmony': 'neuroharmony'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='Harminization, MRI, data science',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_requires=test_requirements,
)

system('pip install git+https://github.com/ncullen93/neuroCombat')
