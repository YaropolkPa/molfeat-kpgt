from setuptools import setup, find_packages

setup(
    name="molfeat-kpgt",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # List dependencies here if needed
    author="Yaropolk Patsahan",
    author_email="yaropolkpa@gmail.com",
    description="Plugin that imports KPGT model into molfeat",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/yourusername/your_package_name",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)