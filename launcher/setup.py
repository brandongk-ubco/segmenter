import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="launcher",
    version="0.0.1",
    author="Brandon Graham-Knight",
    author_email="brandongk@alumni.ubc.ca",
    description="Allows command discovery and launching in multiple adaptors.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brandongk60/segmenter",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: UNLICENSED",
        "Operating System :: Ubuntu :: 18.04",
    ],
    python_requires='>=3.8',
    entry_points={
        "console_scripts": [
            "launch = launcher.launch:launch"
        ]
    }
)
