from setuptools import setup

setup(name='cortex_DIM',
      version='0.12',
      description='The Deep InfoMax package',
      author='R Devon Hjelm',
      author_email='erroneus@gmail.com',
      packages=['cortex_DIM', 'cortex_DIM.configs', 'cortex_DIM.functions', 'cortex_DIM.nn_modules',
                'cortex_DIM.models'],
      install_requires=['cortex==dev0.13'],
      zip_safe=False)
