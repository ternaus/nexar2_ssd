from setuptools import setup, find_packages


setup(
    name='ssd-pytorch',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'torch',
        'torchvision',
    ],
)
