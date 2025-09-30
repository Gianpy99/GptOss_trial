from setuptools import setup, find_packages

setup(
    name='tunix-lora-integration',
    version='0.1.0',
    author='Gianpaolo Borrello',
    author_email='gianpaolo.borrello@gmail.com',
    description='A project for integrating LoRA fine-tuning with Ollama server.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'requests',
        'torch',  # Add any other dependencies required for your project
        'numpy',
        'pandas',
        'pyyaml',
    ],
    entry_points={
        'console_scripts': [
            'tunix-cli=cli:main',  # Adjust this based on your CLI implementation
        ],
    },
)