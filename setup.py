from setuptools import setup, find_packages
import subprocess
import os


def create_virtualenv():
    try:
        subprocess.run(
            [
                "virtualenv",
                "-p",
                "$(which python3.10)",
                "--system-site-packages",
                "football_env",
            ],
            check=True,
            shell=True,
        )
        print("Virtual environment created successfully.")
    except subprocess.CalledProcessError:
        print(
            "Failed to create virtual environment. Please ensure virtualenv is installed."
        )


def install_requirements():
    try:
        subprocess.run(
            ["./football_env/bin/pip", "install", "-r", "requirements.txt"],
            check=True,
            shell=True,
        )
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Failed to install dependencies. Please check the requirements.txt file.")


def setup_directories():
    required_dirs = [
        "input_videos",
        "output_videos",
        "stubs",
        "models",
        "memdump",
        "logs",
    ]
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")
        else:
            print(f"Directory '{directory}' already exists.")


def main():
    print("Setting up directories...")
    setup_directories()

    print("Creating virtual environment...")
    create_virtualenv()

    print("Installing dependencies...")
    install_requirements()


if __name__ == "__main__":
    main()

setup(
    name="Football Analytics Dashboard",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    description="A package for analyzing football games.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/znatri/football-dashboard",
    author="Hardik Goel",
    author_email="hardikkgoel@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "argparse",
        "matplotlib",
        "pandas",
        "supervision",
        "ultralytics",
        "scikit-learn",
        "numpy",
        "opencv-python",
        "colorlog",
        "psutil",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "setup_ml_cv_model=setup:main",
        ],
    },
)
