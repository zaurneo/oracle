# Oracle

## Installation

1. Install Python dependencies (LangChain, yfinance, scikit-learn, etc.):

```bash
pip install -r requirements.txt
```

2. Set required environment variables (either in a `.env` file or your shell):

```bash
gpt_api_key=YOUR_OPENAI_KEY
claude_api_key=YOUR_ANTHROPIC_KEY
# optional search keys
GOOGLE_SEARCH_API_KEY=...
GOOGLE_CUSTOM_SEARCH_ENGINE_ID=...
SERP_API_KEY=...
SEARCH_ENGINE=ddg
```

## Usage

Run the main program:

```bash
python main.py [--full-diagnose]
```

Use `--full-diagnose` to see a detailed diagnostic run.
