import setuptools

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

with open("README.md", "r", encoding="utf-8-sig") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lmdbs",
    version="0.0.1",
    install_requires=requirements,
    author="Mingfu zou",
    author_email="zoumingfu@jwzg.com",
    description="save, read and visualize the data pieces for managing super-large scale mini-piece data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    #packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_namespace_packages(include=["lmdbs", "lmdbs.*"], ),
    include_package_data=True,
)