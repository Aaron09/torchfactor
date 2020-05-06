from setuptools import setup, find_packages
setup(
    name="torchfactor",
    packages=find_packages(),
    version="1.0.0",
    license="MIT",
    description="A package for tensor factorizations using neural networks in PyTorch",
    author="Jonah Casebeer, Aaron Green, Alex Mackowiack",
    author_email="jonahmc2@illinois.edu, aarongg2@illinois.edu, amackow2@illinois.edu",
    url="https://github.com/Aaron09/torchfactor",
    download_url="https://github.com/Aaron09/torchfactor/archive/v1.0.0.tar.gz",
    keywords=["PyTorch", "tensor", "factorization", "neural network", "deep learning"],
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "scikit-image"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.6"
    ],
)
