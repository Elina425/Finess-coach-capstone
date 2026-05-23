# Secure Credential Management Setup

## Quick Start

### 1. Install Required Package
```bash
pip install python-dotenv kaggle
```

Or if using the project requirements:
```bash
pip install -r requirements.txt
```

### 2. Create Your `.env` File

Copy `.env.example` to `.env` and add your actual credentials:

```bash
cp .env.example .env
```

Then edit `.env` with your credentials:
```bash
# Get your Kaggle credentials from: https://www.kaggle.com/settings/account
KAGGLE_USERNAME=your_username_here
KAGGLE_KEY=your_api_key_here
```

### 3. Use in Your Code

**Option A: Automatic setup (recommended)**
```python
from credentials import setup_kaggle_auth

# Automatically authenticates Kaggle API
api = setup_kaggle_auth()

# Now use the API
api.dataset_download_files('dataset-name', path='./data')
```

**Option B: Manual setup**
```python
from credentials import get_kaggle_credentials
import os

username, key = get_kaggle_credentials()
os.environ["KAGGLE_USERNAME"] = username
os.environ["KAGGLE_KEY"] = key

# Now use kaggle CLI or API
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
```

**Option C: Direct environment variable usage**
```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

username = os.getenv("KAGGLE_USERNAME")
key = os.getenv("KAGGLE_KEY")
```

## Security Best Practices

✅ **DO:**
- Store credentials in `.env` (which is in `.gitignore`)
- Regenerate your Kaggle API token if exposed
- Use `get_kaggle_credentials()` from the `credentials.py` helper
- Keep `.env` file locally only, never commit it

❌ **DON'T:**
- Paste API tokens in chat, code, or commit messages
- Commit `.env` to git
- Share credentials in notebooks or documentation
- Use hardcoded API keys

## Regenerating Tokens

If you've exposed your token:
1. Go to https://www.kaggle.com/settings/account
2. Click "Delete" on the existing API token
3. Click "Create New API Token" 
4. Download the new `kaggle.json` or copy the key
5. Update your `.env` file
6. Test with: `python credentials.py`

## Testing Your Setup

```bash
# Test that credentials are loaded correctly
python credentials.py
```

You should see: `✓ Kaggle authentication successful for user: [your_username]`
