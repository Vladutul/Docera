from setuptools import setup, find_packages

setup(
    name='docera',
    version='0.1.0',
    description='A Python library for Intelligent Document Processing.',
    author='Vladutul',
    packages=find_packages(),
    install_requires=[
        'opencv-python>=4.5',
        'Pillow>=9.0',
        'pytesseract>=0.3',
        'rapidfuzz>=2.1',
        'torch>=1.10',
        'transformers>=4.20',
        # Add specific requirements for YoloDetector, if not included in a standard package
        # e.g., 'ultralytics>=8.0' 
    ],
)