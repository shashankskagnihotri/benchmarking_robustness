from setuptools import setup, find_packages


setup(
    name='dispbench',
    version='0.1.0',
    # author='Your Name',
    # author_email='your.email@example.com',
    description='A package for benchmarking the robustness of pixel-wise prediction tasks.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shashankskagnihotri/benchmarking_robustness/',
    packages=find_packages(include=['disparity_estimation*', 'object_detection*']),
    install_requires=[
        'torch',
        'torchvision',
        'mlflow',
        'pandas',
        'PyYAML',
        'pygit2',
        'albumentations',
        'tqdm',
        'tensorboard',
        'natsort',
        'imagecorruptions',
    ],
    license='MIT',
)