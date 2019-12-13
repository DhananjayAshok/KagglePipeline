import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dj_kaggle_pipeline", # Replace with your own username
    version="1.0.3",
    author="Dhananjay Ashok",
    author_email="dhananjay.ashok99@gmail.com",
    description="Pipelines and utility classes for Kaggle and data science!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DhananjayAshok/KagglePipeline",
    download_url = "https://github.com/DhananjayAshok/KagglePipeline/archive/1.0.3.tar.gz",
    packages=setuptools.find_packages(),
    install_requires=[
          'tensorflow',
	  'keras',
	  'matplotlib',
	  'seaborn',
	  'pandas',
	  'numpy',
	  'eli5',
	  'lightgbm',
	  'scikit-learn',
	  'xgboost',

      ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)