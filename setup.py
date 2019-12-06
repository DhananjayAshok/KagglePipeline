import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dj_kaggle_pipeline", # Replace with your own username
    version="0.0.1",
    author="Dhananjay Ashok",
    author_email="dhananjay.ashok99@gmail.com",
    description="Pipelines and utility classes for Kaggle and data science!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)