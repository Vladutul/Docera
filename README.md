# Docera

**Docera** is a Python library for intelligent document processing. It is designed to analyze images of receipts, invoices, or other printed documents, extract relevant information, and structure it for further analysis.

## Features

- **Object Detection**: Detect regions of interest in documents using YOLO.
- **OCR & Text Recognition**: Extract text using Tesseract or the Donut model.
- **Information Refinement**: Use LayoutLMv3 to associate text with semantic labels.
- **Normalization & Classification**: Clean and classify text with a DistilBERT model.
- **Visual Output**: Generate annotated images highlighting detected regions and labels.

## Installation

```bash
git clone https://github.com/yourusername/Docera.git
cd Docera
pip install -r requirements.txt
