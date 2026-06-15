from setuptools import find_namespace_packages, setup

setup(
    name="trisearch_calabi_yau",
    version="0.1.0",
    description="Calabi-Yau triangulation training and sampling tools",
    packages=find_namespace_packages(include=["core*", "data*", "mdp*", "models*"]),
    python_requires=">=3.10",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "torch",
        "torch_geometric",
        "tqdm",
        "wandb",
        "datasets",
        "huggingface_hub",
        "scikit-learn",
    ],
    include_package_data=True,
)
