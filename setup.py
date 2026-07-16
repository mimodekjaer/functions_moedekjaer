import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="modekjar", 
    version="1.0.0",
    author="Mikkel Møller Mødekjær",
    author_email="MikkelMM99@gmail.com",
    description="Just a small function package for numpy, sympy and iminuit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mimodekjaer/functions_moedekjaer",
    project_urls={
        "Bug Tracker": "https://github.com/mimodekjaer/functions_moedekjaer/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.22",
        "matplotlib>=3.5",
        "sympy",
        "iminuit",
    ],
    extras_require={
        "hurtig": ["numba>=0.57"],
    },
)