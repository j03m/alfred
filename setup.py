from setuptools import setup

setup(
    name='machine_learning_finance',
    version="0.2",
    author="Joe Mordetsky",
    author_email="jmordetsky@gmail.com",
    description="All sorts of ml doodads",
    packages=['machine_learning_finance'],
    package_dir={"": "src"},
    tests_require=["pytest"],
    install_requires=[line.rstrip('\n') for line in open('requirements.txt')],
    entry_points={
        "console_scripts": [
            "ppo_train=machine_learning_finance:ppo_train",
        ],
    },
)