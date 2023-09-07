from setuptools import setup, find_packages
import os

root_dir = os.path.dirname(os.path.realpath(__file__))

INSTALL_REQUIRES = [
    "isaacgymenvs",
    "wandb<0.13"
]

setup(
    name="faive_gym",
    author="Soft Robotics Lab",
    version="0.0.1",
    description="IsaacGym environments for the Faive Hand, intended to be used together with IsaacGymEnvs",
    python_requires=">=3.8.*",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
)
