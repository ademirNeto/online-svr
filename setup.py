from setuptools import setup, find_packages

setup(
    name="online-svr",
    version="0.2.0",
    description="A library for making predictions in SVR with online learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ademir Neto",
    author_email="ademir.bsn@gmail.com",
    url="https://github.com/ademirNeto/online-svr",
    packages=find_packages(),
    install_requires=[
    "numpy",
    "pandas",
    "scikit-learn",
    "statsmodels",
    "scipy"
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
