from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="versatile_diffusion", 
    version="1.0.0",
    author="Reese Kneeland",
    description="An installable version of the versatile diffusion package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/reesekneeland/Versatile-Diffusion", 
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
        "matplotlib",
        "pyyaml",
        "numpy",
        "opencv-python",
        "easydict==1.9",
        "scikit-image",
        "tensorboardx==2.1",
        "tensorboard",
        "lpips==0.1.3",
        "fsspec",
        "tqdm",
        "transformers",
        "torchmetrics==1.3.0.post0",
        "einops",
        "omegaconf",
        "open_clip_torch==2.0.2",
        "webdataset",
        "huggingface-hub",
        "gradio==3.17.1",
    ],
    include_package_data=True,
    package_data={
    "versatile_diffusion": ["configs/model/*.yaml"],
    "versatile_diffusion": ["lib/model_zoo/optimus_models/vocab/*.json"],
    "versatile_diffusion": ["lib/model_zoo/optimus_models/vocab/*.txt"],
},
)
