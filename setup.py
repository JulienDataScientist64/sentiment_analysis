from setuptools import setup, find_packages

setup(
    name="sentiment_analysis_project",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)