from setuptools import setup, find_packages


with open("requirements.txt", "r") as f:
    requirements = [
        line.strip()
        for line in f.readlines()
        if not line.startswith("-f")
    ]


with open("README.md", "r") as f:
    long_description = f.read()


def main():
    setup(
        name="cardio-sonix-pipeline",
        version="1.0.0",
        packages=find_packages(),
        include_package_data=True,
        install_requires=requirements,

        license="GNU 3",
        author="cardio-sonix-team",
        author_email="entertomerci@gmail.com",

        description="cardio-sonix-pipeline is a feature processing and "
                    "feature extraction pipeline with a machine learning engine "
                    "that is used by the team cardio-sonix-team to create "
                    "and train machine learning models for heart monitoring",

        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/Cardio-Sonix/cardio-sonix-website",

        classifiers=[
            "Development Status :: 3 - Alpha",
            "License :: GNU 3 GENERAL PUBLIC LICENSE 3",
            "Environment :: Console",
            "Framework :: pytorch",
            "Framework :: pytorch-lightning",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],

        keywords=[
            "ai", "audio",
            "machine learning", "pipeline",
            "pytorch", "pytorch-lightning"
        ],

        python_requires=">=3.10",
    )


if __name__ == "__main__":
    main()
