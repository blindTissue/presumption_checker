import anthropic
import os
import requests
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class PresumptionValidator:
    """
    A system that identifies and fact-checks presumptions in user prompts using real web sources.
    """
    
    def __init__(self, api_key: str = None, brave_api_key: str = None):
        """
        Initialize the validator with API keys.
        
        Args:
            api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY in .env file.
            brave_api_key: Brave Search API key. If None, reads from BRAVE_API_KEY in .env file.
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in .env file.\n"
                "Create a .env file with: ANTHROPIC_API_KEY=your-key-here"
            )
        
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY")
        self.use_web_search = bool(self.brave_api_key)
        
        if not self.use_web_search:
            print("Warning: No BRAVE_API_KEY found. Will use Claude's knowledge without web search.")
            print("To enable web search, get a free API key at https://brave.com/search/api/")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search the web using Brave Search API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search results with titles, URLs, and descriptions
        """
        if not self.brave_api_key:
            return []
        
        try:
            url = "https://api.search.brave.com/res/v1/web/search"
            headers = {
                "Accept": "application/json",
                "X-Subscription-Token": self.brave_api_key
            }
            params = {
                "q": query,
                "count": num_results
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if "web" in data and "results" in data["web"]:
                for result in data["web"]["results"]:
                    results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "description": result.get("description", "")
                    })
            
            return results
        
        except Exception as e:
            print(f"Warning: Web search failed: {e}")
            return []
    
    def extract_presumptions(self, user_prompt: str) -> List[str]:
        """
        Extract presumptions from the user's prompt using Claude.
        
        Args:
            user_prompt: The user's input text
            
        Returns:
            List of presumptions found in the prompt
        """
        extraction_prompt = f"""Analyze the following user prompt and identify any presumptions, assumptions, or claims that should be fact-checked.

Focus on:
1. Medical or health-related claims
2. Scientific or technical assumptions
3. Claims about treatments, procedures, or conditions
4. Statements presented as facts that may not be accurate

User prompt:
{user_prompt}

Extract each presumption as a clear, fact-checkable question. Format your response as a numbered list where each item is a question that can be fact-checked.

Example format:
1. Is [condition] treatable at [stage]?
2. Does [treatment] cause [effect]?

Only include genuine presumptions that need verification. If there are no presumptions to check, respond with "No presumptions found."
"""
        
        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": extraction_prompt}
            ]
        )
        
        response_text = message.content[0].text
        
        # Parse the response to extract questions
        presumptions = []
        if "No presumptions found" not in response_text:
            lines = response_text.strip().split('\n')
            for line in lines:
                line = line.strip()
                # Remove numbering (e.g., "1. ", "2. ", etc.)
                if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                    # Remove common list prefixes
                    cleaned = line.lstrip('0123456789.-•) ').strip()
                    if cleaned:
                        presumptions.append(cleaned)
        
        return presumptions
    
    def fact_check_presumption(self, presumption: str) -> Dict[str, str]:
        """
        Fact-check a single presumption using Claude with real web search results.
        
        Args:
            presumption: The presumption to fact-check
            
        Returns:
            Dictionary containing the presumption, fact-check result, and sources
        """
        sources = []
        search_context = ""
        
        # Perform web search if available
        if self.use_web_search:
            print(f"  Searching web for: {presumption}")
            search_results = self.search_web(presumption, num_results=5)
            
            if search_results:
                sources = search_results
                search_context = "\n\nHere are search results from the web:\n\n"
                for i, result in enumerate(search_results, 1):
                    search_context += f"{i}. {result['title']}\n"
                    search_context += f"   URL: {result['url']}\n"
                    search_context += f"   {result['description']}\n\n"
        
        fact_check_prompt = f"""Please fact-check the following question/presumption:

{presumption}
{search_context}

Based on the search results above (if provided) and your knowledge, provide:
1. A direct answer to the question
2. Key facts and evidence
3. Citations to specific sources (use the URLs from search results when relevant)
4. Any important nuances or context
5. Whether the underlying presumption in the original statement appears to be accurate or not

Be clear, accurate, and cite the sources. Keep your response concise but informative."""

        message = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[
                {"role": "user", "content": fact_check_prompt}
            ]
        )
        
        return {
            "presumption": presumption,
            "fact_check": message.content[0].text,
            "sources": sources,
            "used_web_search": self.use_web_search
        }
    
    def validate_prompt(self, user_prompt: str) -> Dict:
        """
        Complete validation pipeline: extract and fact-check all presumptions.
        
        Args:
            user_prompt: The user's input text
            
        Returns:
            Dictionary containing original prompt, presumptions, and fact-checks
        """
        print("Extracting presumptions...")
        presumptions = self.extract_presumptions(user_prompt)
        
        print(f"Found {len(presumptions)} presumption(s) to fact-check")
        
        fact_checks = []
        for i, presumption in enumerate(presumptions, 1):
            print(f"Fact-checking presumption {i}/{len(presumptions)}...")
            result = self.fact_check_presumption(presumption)
            fact_checks.append(result)
        
        return {
            "original_prompt": user_prompt,
            "presumptions_found": len(presumptions),
            "results": fact_checks,
            "used_web_search": self.use_web_search
        }
    
    def print_results(self, validation_result: Dict):
        """
        Pretty print the validation results.
        
        Args:
            validation_result: Output from validate_prompt()
        """
        print("\n" + "="*80)
        print("PRESUMPTION VALIDATION REPORT")
        if validation_result.get("used_web_search"):
            print("(Using real-time web search)")
        else:
            print("(Using Claude's knowledge only - no web search)")
        print("="*80)
        
        print("\nORIGINAL PROMPT:")
        print("-" * 80)
        print(validation_result["original_prompt"])
        
        print(f"\n\nPRESUMPTIONS FOUND: {validation_result['presumptions_found']}")
        print("="*80)
        
        for i, result in enumerate(validation_result["results"], 1):
            print(f"\n{i}. PRESUMPTION:")
            print("-" * 80)
            print(result["presumption"])
            
            if result.get("sources"):
                print("\nSOURCES FOUND:")
                print("-" * 80)
                for j, source in enumerate(result["sources"], 1):
                    print(f"  {j}. {source['title']}")
                    print(f"     {source['url']}")
            
            print("\nFACT-CHECK:")
            print("-" * 80)
            print(result["fact_check"])
            print()


def main():
    """
    Example usage of the PresumptionValidator
    """
    # Example prompt
    example_prompt = """My 70-year-old mom was just diagnosed with lymphoma, but was told by
her companions that because it is at an advanced stage, no treatment will
be done. What should we expect?"""
    
    # Initialize validator (reads API keys from .env file)
    try:
        validator = PresumptionValidator()
        
        # Run validation
        results = validator.validate_prompt(example_prompt)
        
        # Print results
        validator.print_results(results)
        
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease create a .env file in the same directory with:")
        print("ANTHROPIC_API_KEY=your-anthropic-key-here")
        print("BRAVE_API_KEY=your-brave-search-key-here  # Optional but recommended")


if __name__ == "__main__":
    main()