from setuptools import setup

setup(
    name='regime',
    version='0.0.0.4',
    packages=['regime'],
    url='',
    license='',
    author='bjahnke',
    author_email='bjahnke71@gmail.com',
    description='swing and regime detection functions',
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'pandas_accessors @ git+https://github.com/bjahnke/pandas_accessors.git#egg=pandas_accessors'
    ]
)
