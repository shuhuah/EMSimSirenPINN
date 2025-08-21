from setuptools import setup, find_packages

setup(
    name='your-package-name',
    version='0.1.0',
    description='A short description of your package',
    packages=find_packages(where='src'), # If using src directory
    package_dir={'': 'src'}, # If using src directory
    install_requires=[
        'dependency-package-name',
    ],
)
