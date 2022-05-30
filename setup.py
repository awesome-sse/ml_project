from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='ml in prodaction homework2',
    author='Stanislav',
    license='',
    entry_points={
        "console_scripts": [
            "train_model = src.train_pipeline:train_pipeline_command",
            "predict_model = src.train_pipeline:predict_pipeline_command"
        ]
    },
)
