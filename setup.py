from setuptools import setup, find_packages

setup(
    name='emsimsirenpinn',
    version='0.1.0',
    author='Shuhua Hu',
    author_email='hushuhua37@gmail.com',
    description='2D EM simulation using PINN with SIREN framework',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shuhuah/EMSimSirenPINN',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
