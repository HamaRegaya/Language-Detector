# Language Detector 🌍

A **multi-class text classification** pipeline that identifies 17 languages from text snippets using Multinomial Naive Bayes with Bag-of-Words features.

**Author:** Mohamed Regaya ([HamaRegaya](https://github.com/HamaRegaya))

<img width="1920" height="1656" alt="image" src="https://github.com/user-attachments/assets/cd79427a-518f-4b14-8ac8-5c1e59b3f6e8" />

## Languages Supported

Arabic, Danish, Dutch, English, French, German, Greek, Hindi, Italian, Kannada, Malayalam, Portuguese, Russian, Spanish, Swedish, Tamil, Turkish

## Performance

| Metric | Score |
|--------|-------|
| Accuracy | ~97.8% |
| Weighted F1 | ~97.8% |

## Quick Start

```bash
# Clone
git clone https://github.com/HamaRegaya/Language-Detector.git
cd Language-Detector

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook LanguageDetection.ipynb

# Run tests
pytest tests/ -v
```

## Project Structure

```
Language-Detector/
├── LanguageDetection.ipynb       # Main notebook (training, evaluation, analysis)
├── Language Detection.csv        # Dataset (10k+ samples, 17 languages)
├── trained_pipeline-0.1.0.pkl    # Serialized sklearn pipeline
├── requirements.txt              # Python dependencies
├── tests/
│   └── test_pipeline.py          # Unit tests for pipeline & preprocessing
└── README.md
```

## Pipeline Architecture

```
Text → clean_text() → CountVectorizer (BoW) → MultinomialNB(α=0.1) → Language Label
```

## Dataset

- **Source:** [Kaggle — Language Detection](https://www.kaggle.com/datasets/basilb2s/language-detection) (CC0 / Public Domain)
- **Size:** ~10,337 samples × 2 features (Text, Language)
- **Split:** Stratified 80/20 train/test

## Deployment

The pipeline is serialized as a pickle file and can be served via FastAPI. See the notebook (Section 7) for Dockerfile and API route snippets.

## License

MIT
