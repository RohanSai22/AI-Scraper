# AI Price Comparison Tool

Fetches best prices for any product from various websites using AI-powered web scraping.

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set up environment:

```bash
# Create .env file with:
GROQ_API_KEY=your_groq_api_key_here
```

3. Run with JSON input:

```bash
python main.py '{"country": "US", "query": "iPhone 16 Pro 128GB"}'
```

4. Or run interactive mode:

```bash
python main.py
```

## Input/Output Format

**Input:**

```json
{ "country": "US", "query": "iPhone 16 Pro 128GB" }
```

**Output:**

```json
[
  {
    "link": "https://apple.com/...",
    "price": "999",
    "currency": "USD",
    "productName": "Apple iPhone 16 Pro 128GB"
  }
]
```

## Test Example

```bash
python test_example.py
```

Results are automatically saved to `price_comparison_results.json` after each successful scrape.

## Features

- Multi-country support (US, IN, SG, UK, DE, etc.)
- AI-powered price extraction with browser automation
- Smart URL validation and filtering
- Real-time result saving
- Automatic price sorting
# Assignment
