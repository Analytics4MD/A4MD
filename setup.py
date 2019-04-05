from __future__ import print_function


from setuptools import setup, find_packages

setup(name='a4md',
        version='0.2.3',
        description='Analysis routines for in-situ MD simulations',
        url='https://analytics4md.org',
        author='gclab',
        author_email='stephenthomas2018@gmail.com',
        license='GPL-3.0',
        packages=find_packages(),
        package_dir={'a4md':'a4md'},
        install_requires=[
            'numpy',
            'matplotlib'
        ],
        zip_safe=False)
