---
description: Initial setup for ShopEase project
---

# ShopEase Setup Workflow

This workflow guides you through the initial setup of the ShopEase sentiment analysis project.

## Prerequisites

- Python 3.8 or higher installed
- Git installed (if cloning from repository)
- At least 2GB of free disk space
- Internet connection for downloading dependencies

## Steps

### 1. Clone or Navigate to Repository

If you haven't cloned the repository yet:

```bash
git clone <repository-url>
cd ShopEase
```

If you already have the repository:

```bash
cd ShopEase
```

### 2. Create Virtual Environment

**Windows:**

```bash
python -m venv .venv
```

**Mac/Linux:**

```bash
python3 -m venv .venv
```

### 3. Activate Virtual Environment

**Windows (PowerShell):**

```bash
.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```bash
.venv\Scripts\activate.bat
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

You should see `(.venv)` appear in your terminal prompt.

### 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note:** This may take 5-10 minutes as it installs TensorFlow, PyTorch, and other ML libraries.

### 5. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 6. Download NLTK Data

Run this Python command to download required NLTK data:

```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"
```

### 7. Verify Installation

Test that all key packages are installed:

```bash
python -c "import transformers, torch, tensorflow, spacy, nltk, streamlit; print('✅ All packages installed successfully!')"
```

### 8. Create Required Directories

The following directories will be created automatically when you run the pipeline, but you can create them now:

```bash
mkdir -p data/raw data/processed models/baseline models/bert reports
```

**Windows (PowerShell):**

```bash
New-Item -ItemType Directory -Path "data/raw", "data/processed", "models/baseline", "models/bert", "reports" -Force
```

## Troubleshooting

### Issue: `pip` not found

**Solution:** Make sure Python is added to your PATH, or use `python -m pip` instead of `pip`

### Issue: Permission denied when installing packages

**Solution:** Use `pip install --user -r requirements.txt` or run terminal as administrator

### Issue: spaCy model download fails

**Solution:** Try downloading manually:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl
```

### Issue: TensorFlow/PyTorch installation fails

**Solution:**

- For CPU-only: This should work by default
- For GPU: You may need CUDA toolkit installed
- Try installing one at a time: `pip install torch` then `pip install tensorflow`

### Issue: Virtual environment activation fails on Windows

**Solution:** You may need to enable script execution:

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Verification Checklist

After completing setup, verify:

- [ ] Virtual environment is activated (you see `.venv` in prompt)
- [ ] All packages install without errors
- [ ] spaCy model downloads successfully
- [ ] NLTK data downloads successfully
- [ ] Required directories exist
- [ ] Test import command runs successfully

## Next Steps

Once setup is complete, you can:

1. Run the full pipeline: See `run-pipeline.md` workflow
2. Test predictions: See `test-prediction.md` workflow
3. Launch the dashboard: `streamlit run dashboards/streamlit_app.py`

## Estimated Time

- First-time setup: **15-20 minutes**
- Subsequent setups (with cached downloads): **5 minutes**
