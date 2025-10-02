from setuptools import setup, find_packages

setup(
    name="ml_csv_lib",
    version="0.1.0",
    author="Tu Nombre",
    description="Librería para cargar y preprocesar CSV para ML",
    packages=find_packages(),
    python_requires=">=3.8",
    # NO incluir install_requires - ya están en tu entorno
)