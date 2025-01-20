# BrAIBrook

An iterative transcription system combining DeepSeek's vision capabilities with Kraken's specialized HTR (Handwritten Text Recognition) for historical documents.

## Setup

1. Create and activate virtual environment:
```bash
python3 -m venv kraken_env
source kraken_env/bin/activate
```

2. Install dependencies:
```bash
pip install "kraken[pdf,test,serve]"
```

## Project Structure

```
BrAIBrook/
├── data/
│   └── raw/         # Original manuscript images
├── models/          # Downloaded Kraken models
└── output/          # Processing outputs (binarized images, OCR results)
```

## Current Workflow

1. Download the comprehensive medieval model (8th-16th century):
```bash
kraken get 10.5281/zenodo.7631619
```
This downloads `cremma-generic-1.0.1.mlmodel`

2. Process a manuscript page (example with test_page.jpg):
```bash
# Binarize the image
kraken -i data/raw/test_page.jpg output/test_page_bw.png binarize

# Segment and perform OCR in one step with hOCR output
kraken -h -i output/test_page_bw.png output/test_page.hocr segment ocr -m cremma-generic-1.0.1.mlmodel
```

3. View results:
```bash
# Copy hOCR to HTML for browser viewing
cp output/test_page.hocr output/test_page.html
```
Then open test_page.html in your web browser to view the transcription with positioning information.

## Notes

- Current implementation uses the comprehensive medieval model which covers 8th-16th century manuscripts
- The model is trained on data from GalliCorpora, CREMMA Medieval, and other relevant sources
- hOCR output format provides both transcribed text and positioning information