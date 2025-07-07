import sys
import os
import asyncio
import re
import json
from urllib.parse import urlparse
from datetime import datetime, timedelta
import io
from contextlib import redirect_stdout, redirect_stderr
import logging

# Environment and API
from dotenv import load_dotenv
from browser_use.llm import ChatGoogle
from browser_use import Agent

# Data Handling and Display
import pandas as pd
from IPython.display import display

# Search and Validation
import pycountry
from googlesearch import search
import whois
import ssl
import socket

# --- CONFIGURATION ---
# Validation settings
NUM_SEARCH_RESULTS = 2  # Reduced for faster execution in the scraping phase
MINIMUM_DOMAIN_AGE_DAYS = 365
HIGH_RISK_TLDS = ['.zip', '.top', '.xyz', '.club', '.online', '.loan', '.work', '.gq', '.cf', '.tk']
DOMAIN_BLACKLIST = [
    'youtube.com', 'youtu.be', 'wikipedia.org', 'facebook.com', 'twitter.com', 'instagram.com',
    'pinterest.com', 'linkedin.com', 'reddit.com', 'quora.com', 'google.com', 'amazon.com' # Amazon is often too complex for simple scraping
]
EXCLUSION_KEYWORDS = ['review', 'news', 'vs', 'compare']

# --- HELPER FUNCTIONS FOR VALIDATION ---

def check_domain_age(domain):
    try:
        domain_info = whois.whois(domain)
        creation_date = domain_info.creation_date
        if isinstance(creation_date, list): creation_date = creation_date[0]
        if creation_date is None: return False, "Unknown Age"
        age = (datetime.now() - creation_date).days
        return age > MINIMUM_DOMAIN_AGE_DAYS, f"{age} days old"
    except Exception:
        return False, "WHOIS Error"

def check_url_structure(url_parts):
    if any(url_parts.netloc.endswith(tld) for tld in HIGH_RISK_TLDS):
        return False, "High-Risk TLD"
    if len(url_parts.netloc.split('.')) > 4:
        return False, "Excessive Subdomains"
    return True, "Looks OK"

def check_ssl_certificate(domain):
    try:
        context = ssl.create_default_context()
        with socket.create_connection((domain, 443), timeout=5) as sock:
            with context.wrap_socket(sock, server_hostname=domain) as ssock:
                cert = ssock.getpeercert()
                return bool(cert), "Valid Certificate"
    except Exception:
        return False, "SSL/Socket Error"

# --- CORE LOGIC ---

def get_user_input():
    """Handles Step 1: Interactively gets product and country from the user."""
    print("--- Please provide the product details ---")
    product_query = input("Enter the product you want to search for: ")

    countries_data = sorted([{'name': c.name, 'code': c.alpha_2} for c in pycountry.countries], key=lambda x: x['name'])
    print("\n--- Please select a country from the list below ---")
    for i, country in enumerate(countries_data):
        print(f"{i+1:<4} - {country['name']} ({country['code']})")

    selected_country = None
    while True:
        try:
            choice_str = input(f"\n➡️ Enter the number for your desired country (1-{len(countries_data)}): ")
            choice_num = int(choice_str)
            if 1 <= choice_num <= len(countries_data):
                selected_country = countries_data[choice_num - 1]
                break
            else:
                print(f"❌ Error: Please enter a number between 1 and {len(countries_data)}.")
        except (ValueError, IndexError):
            print("❌ Error: Invalid input. Please enter a valid number.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit()

    logging.info("\n" + "="*50)
    logging.info("✅ INPUT CAPTURED SUCCESSFULLY")
    logging.info("="*50)
    logging.info(f"  -> Product Query: '{product_query}'")
    logging.info(f"  -> Search Location: {selected_country['name']} ({selected_country['code']})")
    return product_query, selected_country['code'], selected_country['name']


def validate_search_results(product_query, country_code):
    """Handles Step 2: Performs Google search and validates URLs."""
    print(f"\n🕵️ Starting search for '{product_query}' in region: {country_code.upper()}...")
    try:
        search_results = search(
            product_query,
            num_results=NUM_SEARCH_RESULTS,
            lang='en',
            region=country_code.lower(),
            advanced=True
        )
        raw_urls = [result.url for result in search_results]
        logging.info(f"✅ Google search found {len(raw_urls)} potential URLs.")
    except Exception as e:
        logging.error(f"\n❌ An error occurred during Google Search: {e}")
        return []

    print(f"\n2. Validating URLs through the Trust Funnel...")
    validated_urls = []
    checked_domains = set()

    for url in raw_urls:
        try:
            url_parts = urlparse(url)
            domain = url_parts.netloc.replace('www.', '')

            if not domain or domain in checked_domains:
                continue
            checked_domains.add(domain)

            # --- RUNNING THE FUNNEL ---
            if any(blacklisted in domain for blacklisted in DOMAIN_BLACKLIST):
                continue
            if any(keyword in url.lower() for keyword in EXCLUSION_KEYWORDS):
                continue
            if not url.startswith('https://'):
                continue

            is_structured_ok, _ = check_url_structure(url_parts)
            if not is_structured_ok:
                continue

            is_old_enough, _ = check_domain_age(domain)
            if not is_old_enough:
                continue
            
            is_cert_valid, _ = check_ssl_certificate(domain)
            if not is_cert_valid:
                continue

            # If all checks pass, add to the list
            validated_urls.append(url)
        except Exception:
            continue
            
    logging.info(f"✅ Validation complete. Found {len(validated_urls)} high-trust URLs.")
    return validated_urls


async def analyze_log_with_llm(terminal_log: str, product_query: str, llm: ChatGoogle):
    """
    Uses an LLM to analyze the raw terminal output from the agent to determine
    the final price and availability.
    """
    print("🧠 Sending terminal log to Gemini for final analysis...")
    
    analysis_prompt = f"""
    You are a data analyst. Your task is to analyze the provided terminal log from a web scraping agent.
    The agent's goal was to find the price for the product: "{product_query}".

    Based on the log, determine the final outcome. Answer in a JSON format with two keys: 'price' and 'status'.
    - 'price': Should be a float number (e.g., 1399.00). If the price is not found or the product is unavailable, set it to null.
    - 'status': Should be a string. Possible values are "Available", "Out of Stock", "Not Found", or "Scraping Error".

    Look for patterns like:
    - "📄 Result: 119900" or "📄 Result: 107900" (extract the number)
    - "✅ Task completed successfully" (indicates success)
    - "Out of Stock" or "Not Found" (indicates unavailable)
    - Any price extraction or currency symbols

    Here is the terminal log:
    --- LOG START ---
    {terminal_log}
    --- LOG END ---

    Provide only the JSON object in your response.
    """
    
    try:
        # Use the correct method for ChatGoogle
        messages = [{"role": "user", "content": analysis_prompt}]
        response = await llm.achat(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Clean the response to get only the JSON part
        json_match = re.search(r'```json\n(.*?)```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response_text.strip()
            
        result = json.loads(json_str)
        print(f"✅ Gemini analysis complete: {result}")
        return result
    except Exception as e:
        print(f"❌ Error during Gemini log analysis: {e}")
        return {"price": None, "status": "Analysis Error"}


async def scrape_prices_with_agent(product_query, urls, llm):
    """Handles Step 3: Iterates through URLs, uses the AI agent to get the price, and logs everything."""
    scraped_data = []
    total_urls = len(urls)
    MODEL_OUTPUT_DIR = os.path.join("logs", "model_output")
    COMPLETE_VIEW_DIR = os.path.join("logs", "complete_view")
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    os.makedirs(COMPLETE_VIEW_DIR, exist_ok=True)

    for i, url in enumerate(urls, 1):
        logging.info("\n" + "-"*60)
        logging.info(f"🔎 Processing URL {i}/{total_urls}: {url}")

        # Limit the sanitized URL length to avoid FileNotFoundError on Windows
        sanitized_url = re.sub(r'https?://(www\.)?', '', url).replace('/', '_').replace(':', '_').replace('?', '_').replace('=', '_').replace('&', '_')[:100]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_base_name = f"{sanitized_url}_{timestamp}"
        model_output_log_path = os.path.join(MODEL_OUTPUT_DIR, f"{log_base_name}.log")
        complete_view_log_path = os.path.join(COMPLETE_VIEW_DIR, f"{log_base_name}.log")

        task = f"""
        Go to the website: {url}
        Your task is to find the price for the product: "{product_query}".
        1. Navigate the page. If there are product options (e.g., color, storage, model), select the ones that best match the query.
        2. Find the final price.
        3. If the product is out of stock, unavailable, or a price cannot be found, state that clearly.
        Your final response must be concise. Return ONLY the price as a number (e.g., '1099.99') OR the status 'Out of Stock' or 'Not Found'.
        Do not include currency symbols or any other text.
        """

        # --- Logging Setup ---
        log_capture_stream = io.StringIO()
        agent_logger = logging.getLogger('browser_use')
        stream_handler = logging.StreamHandler(log_capture_stream)
        original_handlers = agent_logger.handlers[:]
        original_level = agent_logger.level
        agent_logger.addHandler(stream_handler)
        agent_logger.setLevel(logging.INFO)

        terminal_output = ""
        raw_agent_output = None
        agent_result_obj = None
        
        try:
            agent = Agent(task=task, llm=llm, max_loops=8)
            agent_result_obj = await agent.run()

            # Extract the raw final answer from the agent's result
            if agent_result_obj:
                # Try to get the final result from the agent
                if hasattr(agent_result_obj, 'final_result'):
                    raw_agent_output = agent_result_obj.final_result
                elif hasattr(agent_result_obj, 'result'):
                    raw_agent_output = agent_result_obj.result
                elif hasattr(agent_result_obj, 'last_result'):
                    raw_agent_output = agent_result_obj.last_result
                # Check if it's a list-like object with results
                elif hasattr(agent_result_obj, '__iter__') and not isinstance(agent_result_obj, str):
                    try:
                        # Look for the last result in the history
                        results = list(agent_result_obj)
                        if results:
                            for result in reversed(results):
                                if hasattr(result, 'extracted_content'):
                                    raw_agent_output = result.extracted_content
                                    break
                                elif hasattr(result, 'long_term_memory'):
                                    if 'Result:' in str(result.long_term_memory):
                                        raw_agent_output = result.long_term_memory
                                        break
                    except:
                        pass
                else:
                    raw_agent_output = str(agent_result_obj)
            else:
                raw_agent_output = "No result returned"
            
            print(f"✅ Agent task finished. Raw Output: '{raw_agent_output}'")

        except Exception as e:
            print(f"❌ Agent execution failed for {url}: {e}")
            log_capture_stream.write(f"\n--- AGENT EXECUTION FAILED ---\nURL: {url}\nERROR: {e}\n")
            raw_agent_output = f"Agent Execution Error: {e}"
        finally:
            # --- Restore logger and get captured output ---
            agent_logger.handlers[:] = original_handlers
            agent_logger.setLevel(original_level)
            terminal_output = log_capture_stream.getvalue()
            
            # --- Save logs regardless of outcome ---
            with open(model_output_log_path, 'w', encoding='utf-8') as f:
                f.write(terminal_output)
            print(f"📄 Verbose agent logs saved to: {model_output_log_path}")
            
            if agent_result_obj:
                with open(complete_view_log_path, 'w', encoding='utf-8') as f:
                    f.write(str(agent_result_obj))
                print(f"📄 Complete agent history saved to: {complete_view_log_path}")

        # --- Analyze the captured log with the LLM ---
        analysis_result = await analyze_log_with_llm(terminal_output, product_query, llm)

        price = analysis_result.get('price')
        status = analysis_result.get('status', 'Analysis Error')

        # Fallback: Try to extract price from terminal log directly
        if price is None:
            # Look for "📄 Result: [number]" pattern
            result_match = re.search(r'📄 Result: (\d+(?:\.\d+)?)', terminal_output)
            if result_match:
                try:
                    price = float(result_match.group(1))
                    status = "Available"
                    print(f"🔍 Extracted price from terminal log: {price}")
                except:
                    pass
            
            # Look for "✅ Task completed successfully" with a result
            elif "✅ Task completed successfully" in terminal_output:
                # Check if raw_agent_output is numeric
                if raw_agent_output and isinstance(raw_agent_output, (int, float)):
                    price = float(raw_agent_output)
                    status = "Available"
                elif raw_agent_output and str(raw_agent_output).replace('.', '').replace(',', '').isdigit():
                    price = float(str(raw_agent_output).replace(',', ''))
                    status = "Available"
                elif "Not Found" in terminal_output:
                    status = "Not Found"
                elif "Out of Stock" in terminal_output:
                    status = "Out of Stock"
                else:
                    status = "Scraping Error"
            else:
                status = "Scraping Error"

        # Determine currency from the log
        currency = "USD"  # default
        if "₹" in terminal_output or "INR" in terminal_output.upper():
            currency = "INR"
        elif "£" in terminal_output or "GBP" in terminal_output.upper():
            currency = "GBP"
        elif "€" in terminal_output or "EUR" in terminal_output.upper():
            currency = "EUR"

        scraped_data.append({
            "url": url,
            "price": float(price) if price is not None else None,
            "status": status,
            "raw_output": raw_agent_output,
            "terminal_log": terminal_output,
            "currency": currency
        })
            
    return scraped_data


def display_final_results(data, country_name):
    """Sorts and displays the final scraped data in a table and JSON format."""
    if not data:
        print("\nNo price data could be collected.")
        return

    # Sort results by price, ascending. None prices go to the bottom.
    sorted_data = sorted(data, key=lambda x: x['price'] if x['price'] is not None else float('inf'))

    # --- Display as a Table --
    print("\n" + "="*80)
    print(f"🛒 FINAL PRICE COMPARISON REPORT (Location: {country_name})")
    print("="*80)
    
    display_list = []
    for item in sorted_data:
        if item['price'] is not None:
            currency_symbol = "₹" if item.get('currency') == "INR" else "$"
            price_display = f"{currency_symbol}{item['price']:.2f}"
        else:
            price_display = "N/A"
        
        display_list.append({
            "Retailer URL": item['url'],
            "Price": price_display,
            "Availability": item['status'],
        })
    
    df = pd.DataFrame(display_list)
    df.set_index('Retailer URL', inplace=True)
    try:
        # Try to use IPython display for rich output in notebooks
        display(df)
    except Exception:
        # Fallback for standard Python terminals
        print(df.to_string())

    # --- Display as Detailed Text (Full Details) ---
    print("\n" + "="*80)
    print("📋 DETAILED TEXT OUTPUT (per URL)")
    print("="*80)
    for item in sorted_data:
        if item['price'] is not None:
            currency_symbol = "₹" if item.get('currency') == "INR" else "$"
            price_display = f"{currency_symbol}{item['price']:.2f}"
        else:
            price_display = "N/A"
            
        print(f"URL: {item['url']}")
        print(f"  - Price: {price_display}")
        print(f"  - Availability: {item['status']}")
        print(f"  - Raw Agent Output: {item.get('raw_output', 'N/A')}")
        print(f"  - Currency: {item.get('currency', 'N/A')}")
        print("-"*60)

    # --- Display as JSON ---
    print("\n" + "="*80)
    print("📋 JSON OUTPUT (full data)")
    print("="*80)
    
    json_list = []
    for item in sorted_data:
        if item['price'] is not None:
            currency_symbol = "₹" if item.get('currency') == "INR" else "$"
            price_display = f"{currency_symbol}{item['price']:.2f}"
        else:
            price_display = "N/A"
            
        json_list.append({
            "Retailer URL": item['url'],
            "Price": price_display,
            "Availability": item['status'],
            "Raw Output": str(item.get('raw_output', '')),
            "Currency": item.get('currency', 'N/A'),
            "Scraped Successfully": item['price'] is not None
        })
    print(json.dumps(json_list, indent=2, ensure_ascii=False))


async def main():
    """Main asynchronous function to run the entire workflow."""
    # --- Setup logging ---
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # --- Create Log Directories ---
    LOGS_DIR = "logs"
    os.makedirs(LOGS_DIR, exist_ok=True)
    logging.info(f"📂 Log directory ensured at: ./{LOGS_DIR}/")

    # --- Load Environment and Check API Key ---
    load_dotenv()
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key or gemini_api_key == "YOUR_GEMINI_API_KEY":
        logging.error("❌ CRITICAL ERROR: GEMINI_API_KEY is not set.")
        logging.error("   Please create a .env file and add your key: GEMINI_API_KEY='...'")
        sys.exit(1)

    # --- Step 1: Get User Input ---
    product_query, country_code, country_name = get_user_input()
    
    # --- Step 2: Validate Search Results ---
    validated_urls = validate_search_results(product_query, country_code)

    if not validated_urls:
        logging.info("\nNo credible URLs found. Exiting.")
        return

    logging.info("\n--- Final List of High-Trust URLs to Scrape ---")
    for u in validated_urls:
        logging.info(f"  -> {u}")

    # --- Step 3: Scrape Prices with AI Agent ---
    logging.info("\n" + "#"*80)
    logging.info("🤖 INITIALIZING AI BROWSER AGENT FOR PRICE SCRAPING")
    logging.info("#"*80)
    
    llm = ChatGoogle(model="gemini-2.5-flash", api_key=gemini_api_key)
    scraped_data = await scrape_prices_with_agent(product_query, validated_urls, llm)
    
    # --- Final Step: Display Sorted Results ---
    display_final_results(scraped_data, country_name)
    
    # --- Save complete run log ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_log_path = os.path.join(LOGS_DIR, f"run_{timestamp}.log")
    with open(run_log_path, 'w', encoding='utf-8') as f:
        f.write(f"Product Query: {product_query}\n")
        f.write(f"Country: {country_name} ({country_code})\n")
        f.write(f"URLs Processed: {len(validated_urls)}\n")
        f.write(f"Results:\n")
        for item in scraped_data:
            if item['price'] is not None:
                currency_symbol = "₹" if item.get('currency') == "INR" else "$"
                price_str = f"{currency_symbol}{item['price']:.2f}"
            else:
                price_str = "N/A"
            f.write(f"  - {item['url']}: {price_str} ({item['status']})\n")
    
    logging.info(f"✅ Script finished. Full log saved to {run_log_path}")
    print(f"✅ Script finished. Full log saved to {run_log_path}")


if __name__ == "__main__":
    # Ensure asyncio event loop is managed correctly
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Exiting.")