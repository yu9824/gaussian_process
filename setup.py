from setuptools import setup, find_packages

with open('requirements.txt') as requirements_file:
    install_requirements = requirements_file.read().splitlines()

setup(
    name='gaussian_process',
    version='0.0.6',
    description='',
    author='yu-9824',
    author_email='yu.9824{0}gmail.com'.format('@'),
    install_requires=install_requirements,
    url='https://github.com/yu-9824/gaussian_process',
    license=license,
    packages=find_packages(exclude=['example'])
)
