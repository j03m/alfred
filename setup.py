from setuptools import setup

setup(
    name='alfred',
    version="0.2",
    author="Joe Mordetsky",
    author_email="jmordetsky@gmail.com",
    description="All sorts of ml doodads",
    packages=['alfred'],
    package_dir={"": "src"},
    tests_require=["pytest"],
    install_requires=[line.rstrip('\n') for line in open('requirements.txt')],
    entry_points={
        "console_scripts": [
            "train_eval_all_models=alfred:train_eval_all_models",
        ],
    },
)