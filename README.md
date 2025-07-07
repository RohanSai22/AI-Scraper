# AI Web Scraper

An advanced AI-powered web scraper that finds and compares product prices across multiple retailers using browser automation and Gemini AI.

## Features

- 🔍 Smart Google product search
- 🛡️ Multi-layer trust validation (domain age, SSL, blacklist)
- 🤖 AI-powered scraping (Gemini 2.5 Flash)
- 🌍 Multi-country & multi-currency support
- 📊 Table & JSON output, full logging

## Why This Approach?

- Other LLMs (Groq+Llama3, Mavericka, Scout) hit context limits (500k tokens) and fail on large logs
- Heuristic/selenium fallback is less robust for dynamic sites
- `browser_use` + Gemini gives best reliability for price extraction

## Requirements

- Python 3.11+
- Google Gemini API key
- Internet connection

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/RohanSai22/AI-Scraper.git
   cd AI-Scraper
   ```

2. **Create a Python virtual environment (Python 3.11+ recommended):**

   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**

   ```bash
   uv pip install -r requirements.txt
   ```

4. **Install Playwright browsers:**

   ```bash
   playwright install
   ```

5. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

## Usage

Run:

```bash
python main.py
```

Follow the prompts for product and country. Results are shown in table and JSON formats.

## Output Example

```
🛒 FINAL PRICE COMPARISON REPORT (Location: India)
Retailer URL                                 Price        Availability
https://www.flipkart.com/...                 ₹107900.00   Available
https://www.apple.com/in/shop/...            ₹119900.00   Available
```

## Notes

- Works with most e-commerce sites seamlessly (Apple, Flipkart, BestBuy, etc.)
- All logs saved in `logs/` directory
- For issues, check logs and your API key

## License

MIT License. For research/educational use only. Respect site terms and robots.txt.
