import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchext",
    version="0.0.3",
    packages=setuptools.find_packages(),

    install_requires=[
        "tensorboardX",
        "PyYAML",
    ],

    ptrhon_requires=">=3.5",

    author="Yongjin Cho",
    author_email="yongjin.cho@gmail.com",
    description="A PyTorch extension library for easy experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    url="https://github.com/yongjincho/torchext",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
