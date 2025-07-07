# AI Web Scraper


A fast, robust AI-powered product price scraper using browser_use + Gemini. Finds, validates, and extracts prices from top e-commerce sites with multi-country and multi-currency support.

## SAMPLE OUTPUT :
![image](https://github.com/user-attachments/assets/66cbec89-be6b-4612-9ab1-ecbec2cc1d83)


---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/RohanSai22/AI-Scraper.git
   cd "AI-Scraper"
   ```

2. **Create a Python 3.11+ virtual environment:**

   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies (use uv for speed):**

   ```bash
   pip install uv
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

---

## Usage

Run:

```bash
python main.py
```

- Enter your product and country when prompted.
- Results are shown in table, text, and JSON formats.

---

## Notes & Tips

- **Google Search 429 Error:**
  > ❌ An error occurred during Google Search: 429 Client Error: Too Many Requests ...
  - This is due to rate-limiting by Google (since we use `googlesearch-python` as a workaround for not using a paid SERP API). If you see this, just retry (it usually works after 2-4 tries).

- **URL Limit:**
  - By default, only 2 URLs are scraped per search for speed and to avoid Gemini API quota issues. You can increase this in `main.py` (`NUM_SEARCH_RESULTS`), but be aware of API usage if on a free tier.

- **Why browser_use + Gemini?**
  - Other approaches (Groq+Llama3, Mavericka, Scout, Selenium+heuristics) were tested but failed on context size, reliability, or dynamic content. 
    
    browser_use with Gemini along {gemini-2.5-flash} model is the most robust for real-world e-commerce scraping.We can try it out with gemini-2.5-pro as well.

---

## Features

- Google search + trust validation
- AI browser agent for price extraction
- Multi-country, multi-currency
- Table, text, and JSON output
- Full logs in `logs/`

---

## Troubleshooting

- Use a valid Gemini API key
- Use a fresh virtual environment
- If scraping fails, retry or increase `NUM_SEARCH_RESULTS`

---

MIT License
