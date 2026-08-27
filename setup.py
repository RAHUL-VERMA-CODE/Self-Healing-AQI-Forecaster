from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT="-e ."

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    try:
        with open(file_path,"r")as file:
            requirements=file.readlines()
            requirements=[req.strip() for req in requirements]

            if HYPHEN_E_DOT in requirements:
                requirements.remove(HYPHEN_E_DOT)

    except FileNotFoundError:
        print(f"{file_path} not found.")

    return requirements

setup(
    name="Self-Healing AQI Forecaster",
    version="0.0.1",
    author="Rahul verma",
    author_email="rahulverma96259@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")

)

                