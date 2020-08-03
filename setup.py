from distutils.core import setup

setup(
    name='stars',
    version='0.0.1',
    author='Juan L Gamella',
    author_email='juangamella@gmail.com',
    packages=['stars', 'stars.test'],
    scripts=[],
    url='http://pypi.python.org/pypi/stars/',
    license='LICENSE.txt',
    description='Implementation of the StARS algorithm for regularization selection',
    long_description=open('README.txt').read(),
    install_requires=['numpy', 'sklearn>=0.20']
)
