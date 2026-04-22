# Hassaniya Chatbot — مبارك ولد حميدة

A simple Arabic Hassaniya chatbot built with Streamlit and Hugging Face Transformers.

## About

This repository hosts a conversational app for a Hassaniya dialect model named **مبارك ولد حميدة**. It can run it by downloading the model artifact automatically from Weights & Biases.

The app is designed for Arabic dialect chat in a polished Streamlit UI with right-to-left layout and custom styling.

## Features

- Hassaniya dialect conversational model
- Streamlit-based chat interface
- Local or remote model loading
- Custom Arabic UI styling

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

3. Open the local Streamlit URL shown in your terminal.


## Files

- `app.py` — Streamlit application and model inference logic
- `requirements.txt` — required Python packages
- `data/train.json` — training dataset example
- `data/test.json` — test dataset example

## Notes

- The model prompt format uses Arabic labels: `سؤال:` and `جواب:`.
- `MODEL_PATH` can be overridden via the environment variable for local or custom model directories.

## Usage example

**Example input:**

```text
سؤال: 
الوركة موجود؟
```

**Example response:**

```text
عندي حت ، شتبقي منه ربي يحفظك
```
