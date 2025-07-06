from browser_use.llm import ChatGroq
from browser_use import Agent
from dotenv import load_dotenv
import asyncio
import json
import pandas as pd
import pycountry
import ssl
import socket
from urllib.parse import urlparse
from datetime import datetime
from googlesearch import search
import whois
import re

load_dotenv()

class PriceComparator:
    def __init__(self):
        self.llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")
        self.num_results = 10
        self.domain_blacklist = [
            'youtube.com', 'youtu.be', 'wikipedia.org', 'facebook.com', 'twitter.com', 
            'instagram.com', 'pinterest.com', 'linkedin.com', 'reddit.com', 'quora.com'
        ]
        self.high_risk_tlds = ['.zip', '.top', '.xyz', '.club', '.online', '.loan', '.work']
        
    def get_user_input(self):
        """Get product query and country from user"""
        product_query = input("Enter the product you want to search for: ")
        
        # Get countries data
        countries_data = sorted([{'name': c.name, 'code': c.alpha_2} for c in pycountry.countries], 
                               key=lambda x: x['name'])
        
        print("\n--- Select a country ---")
        for i, country in enumerate(countries_data[:20]):  # Show first 20 for brevity
            print(f"{i+1:<3} - {country['name']} ({country['code']})")
        print("... (or enter country code directly)")
        
        # Get country selection
        while True:
            try:
                choice = input(f"\nEnter number (1-20) or country code (e.g., US, IN, SG): ").strip()
                
                if choice.isdigit() and 1 <= int(choice) <= 20:
                    selected_country = countries_data[int(choice) - 1]
                    break
                elif len(choice) == 2 and choice.upper() in [c['code'] for c in countries_data]:
                    selected_country = next(c for c in countries_data if c['code'] == choice.upper())
                    break
                else:
                    print("❌ Invalid input. Try again.")
            except KeyboardInterrupt:
                print("\nOperation cancelled.")
                return None, None
        
        return product_query, selected_country
    
    def validate_domain(self, domain):
        """Quick domain validation"""
        try:
            # Check blacklist
            if any(blacklisted in domain for blacklisted in self.domain_blacklist):
                return False, "Blacklisted domain"
            
            # Check TLD
            if any(domain.endswith(tld) for tld in self.high_risk_tlds):
                return False, "High-risk TLD"
            
            # Check SSL (quick check)
            try:
                context = ssl.create_default_context()
                with socket.create_connection((domain, 443), timeout=3) as sock:
                    with context.wrap_socket(sock, server_hostname=domain) as ssock:
                        return True, "Valid"
            except:
                return False, "SSL issues"
                
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def search_and_validate_urls(self, product_query, country_code):
        """Search for URLs and validate them"""
        print(f"\n🔍 Searching for '{product_query}' in {country_code}...")
        
        try:
            # Add some delay to avoid rate limiting
            import time
            time.sleep(2)
            
            # Try multiple search strategies
            search_queries = [
                f"{product_query} buy online",
                f"{product_query} price",
                f"{product_query} shop"
            ]
            
            raw_urls = []
            for query in search_queries:
                try:
                    # Perform Google search
                    search_results = search(
                        query,
                        num_results=self.num_results // len(search_queries),
                        lang='en',
                        region=country_code.lower(),
                        advanced=True
                    )
                    
                    for result in search_results:
                        raw_urls.append(result.url)
                    
                    time.sleep(1)  # Delay between searches
                    
                except Exception as e:
                    print(f"❌ Search error for query '{query}': {e}")
                    continue
            
            print(f"✅ Found {len(raw_urls)} URLs")
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            # Fallback: use some known shopping sites
            raw_urls = [
                f"https://www.apple.com/search/{product_query.replace(' ', '%20')}",
                f"https://www.amazon.com/s?k={product_query.replace(' ', '+')}",
                f"https://www.bestbuy.com/site/searchpage.jsp?st={product_query.replace(' ', '%20')}",
                f"https://www.walmart.com/search?q={product_query.replace(' ', '%20')}",
                f"https://www.target.com/s?searchTerm={product_query.replace(' ', '%20')}"
            ]
            print(f"✅ Using fallback URLs: {len(raw_urls)} URLs")
        
        # Validate URLs
        validated_urls = []
        checked_domains = set()
        
        for url in raw_urls:
            try:
                url_parts = urlparse(url)
                domain = url_parts.netloc.replace('www.', '')
                
                if domain in checked_domains or not domain:
                    continue
                    
                checked_domains.add(domain)
                is_valid, reason = self.validate_domain(domain)
                
                if is_valid:
                    validated_urls.append(url)
                    print(f"✅ {domain}")
                else:
                    print(f"❌ {domain} - {reason}")
                    
            except Exception as e:
                print(f"❌ Error validating {url}: {e}")
        
        print(f"\n🎯 Final validated URLs: {len(validated_urls)}")
        return validated_urls
    
    def split_text_for_llm(self, text, max_tokens=5000):
        """Split text to stay within token limits (rough estimate: 4 chars = 1 token)"""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return [text]
        
        # Split into chunks
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            # Try to split at word boundaries
            if end < len(text):
                # Find last space before the limit
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            chunks.append(text[start:end])
            start = end
        
        return chunks

    async def scrape_price_from_url(self, url, product_query):
        """Scrape price from a single URL using browser automation"""
        try:
            # Create a focused task that respects token limits
            task = f"""
            Go to: {url}
            
            Search for product: {product_query}
            
            Extract ONLY:
            1. Product name (exact title)
            2. Price (numbers only, no symbols)
            3. Currency (USD/INR/SGD/etc.)
            
            Return JSON:
            {{"productName": "name", "price": "999", "currency": "USD"}}
            
            If no price: {{"error": "No price found"}}
            """
            
            # Ensure task is within token limits
            if len(task) > 5000 * 4:  # 5000 tokens * 4 chars per token
                task = f"""
                Go to: {url}
                Find: {product_query}
                Extract: name, price, currency
                JSON: {{"productName": "name", "price": "999", "currency": "USD"}}
                """
            
            agent = Agent(task=task, llm=self.llm)
            result = await agent.run()
            
            # Try to extract JSON from the result
            try:
                # Look for JSON in the response
                json_match = re.search(r'\{.*\}', str(result), re.DOTALL)
                if json_match:
                    price_data = json.loads(json_match.group())
                    if 'error' not in price_data and 'price' in price_data:
                        price_data['link'] = url
                        return price_data
            except Exception as json_error:
                print(f"❌ JSON parsing error: {json_error}")
            
            return None
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return None
    
    async def scrape_all_prices(self, urls, product_query):
        """Scrape prices from all URLs"""
        print(f"\n💰 Scraping prices from {len(urls)} URLs...")
        
        results = []
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] Scraping: {url}")
            price_data = await self.scrape_price_from_url(url, product_query)
            
            if price_data:
                results.append(price_data)
                print(f"✅ Found: {price_data.get('productName', 'N/A')} - {price_data.get('price', 'N/A')} {price_data.get('currency', 'N/A')}")
                
                # Save result immediately after each successful scrape
                self.save_results_incrementally(results)
            else:
                print("❌ No price found")
        
        return results
    
    def sort_and_display_results(self, results):
        """Sort results by price and display"""
        if not results:
            print("\n❌ No prices found!")
            return []
        
        # Sort by price (convert to float for proper sorting)
        try:
            valid_results = []
            for result in results:
                try:
                    price = float(re.sub(r'[^\d.]', '', str(result.get('price', '0'))))
                    result['price_numeric'] = price
                    valid_results.append(result)
                except:
                    continue
            
            sorted_results = sorted(valid_results, key=lambda x: x['price_numeric'])
            
            # Display as table
            print("\n" + "="*80)
            print("🏆 PRICE COMPARISON RESULTS (Sorted by Price)")
            print("="*80)
            
            df_data = []
            for result in sorted_results:
                df_data.append({
                    'Product': result.get('productName', 'N/A')[:50],
                    'Price': result.get('price', 'N/A'),
                    'Currency': result.get('currency', 'N/A'),
                    'Website': urlparse(result.get('link', '')).netloc
                })
            
            df = pd.DataFrame(df_data)
            print(df.to_string(index=False))
            
            # Format for JSON output
            final_results = []
            for result in sorted_results:
                final_results.append({
                    'link': result.get('link', ''),
                    'price': str(result.get('price', '')),
                    'currency': result.get('currency', ''),
                    'productName': result.get('productName', '')
                })
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error sorting results: {e}")
            return results
    
    def save_results_incrementally(self, results):
        """Save results to JSON file after each successful scrape"""
        try:
            # Sort results by price before saving
            valid_results = []
            for result in results:
                try:
                    price = float(re.sub(r'[^\d.]', '', str(result.get('price', '0'))))
                    result['price_numeric'] = price
                    valid_results.append(result)
                except:
                    continue
            
            sorted_results = sorted(valid_results, key=lambda x: x['price_numeric'])
            
            # Format for JSON output
            final_results = []
            for result in sorted_results:
                final_results.append({
                    'link': result.get('link', ''),
                    'price': str(result.get('price', '')),
                    'currency': result.get('currency', ''),
                    'productName': result.get('productName', '')
                })
            
            # Save to JSON file
            output_file = 'price_comparison_results.json'
            with open(output_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            print(f"💾 Results saved to {output_file} ({len(final_results)} items)")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")
    
    async def run(self, product_query=None, country_code=None):
        """Main execution function"""
        print("🛒 AI Price Comparison Tool")
        print("="*50)
        
        # Get input
        if not product_query or not country_code:
            product_query, country_data = self.get_user_input()
            if not product_query:
                return
            country_code = country_data['code']
            country_name = country_data['name']
        else:
            country_name = country_code
        
        print(f"\n🎯 Searching for: {product_query}")
        print(f"📍 Location: {country_name} ({country_code})")
        
        # Search and validate URLs
        validated_urls = self.search_and_validate_urls(product_query, country_code)
        
        if not validated_urls:
            print("❌ No valid URLs found!")
            return
        
        # Scrape prices
        results = await self.scrape_all_prices(validated_urls[:5], product_query)  # Limit to 5 for speed
        
        # Sort and display results
        final_results = self.sort_and_display_results(results)
        
        # Save results to JSON
        output_file = 'price_comparison_results.json'
        with open(output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n💾 Results saved to {output_file}")
        
        return final_results

async def main():
    """Main function for direct execution"""
    import sys
    
    # Check if JSON input is provided as command line argument
    if len(sys.argv) > 1:
        try:
            input_data = json.loads(sys.argv[1])
            product_query = input_data.get('query')
            country_code = input_data.get('country')
            
            if not product_query or not country_code:
                print("❌ Invalid JSON input. Expected: {'query': 'product', 'country': 'US'}")
                return
                
            comparator = PriceComparator()
            results = await comparator.run(product_query, country_code)
            
            # Output results as JSON
            if results:
                print(json.dumps(results, indent=2))
            else:
                print("[]")
                
        except json.JSONDecodeError:
            print("❌ Invalid JSON input")
    else:
        # Interactive mode
        comparator = PriceComparator()
        await comparator.run()

if __name__ == "__main__":
    asyncio.run(main())