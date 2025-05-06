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
    packages=find_packages(),
    install_requires=[
        'torch==2.6.0',
        'torchvision==0.21.0',
        'mlflow==2.21.3',
        'pandas==2.2.3',
        'PyYAML==6.0.2',
        'pygit2==1.17.0',
        'albumentations==2.0.5',
        'tqdm==4.67.1',
        'tensorboard==2.19.0',
        'natsort==8.4.0',
        'imagecorruptions==1.1.2',
    ],
    license='MIT',
)