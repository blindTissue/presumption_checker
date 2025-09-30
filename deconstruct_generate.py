import anthropic
import os
from typing import List, Dict
from dotenv import load_dotenv
import datasets

load_dotenv()

class PresumptionValidator:
    
    def __init__(self, api_key: str = None, model: str = "claude-3-5-haiku-20241022"):

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided or set in .env file.\n"
            )
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = model
    
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
            model=self.model,
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
        Fact-check a single presumption using Claude with web search capability.
        
        Args:
            presumption: The presumption to fact-check
            
        Returns:
            Dictionary containing the presumption and fact-check result
        """
        fact_check_prompt = f"""Please fact-check the following question/presumption:

{presumption}

Provide:
1. A direct answer to the question
2. Key facts and evidence
3. Any important nuances or context
4. Whether the underlying presumption in the original statement appears to be accurate or not

Be clear and accurate."""

        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": fact_check_prompt}
            ]
        )
        
        return {
            "presumption": presumption,
            "fact_check": message.content[0].text
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
            "results": fact_checks
        }
    
    def print_results(self, validation_result: Dict):
        """
        Pretty print the validation results.
        
        Args:
            validation_result: Output from validate_prompt()
        """
        print("\n" + "="*80)
        print("PRESUMPTION VALIDATION REPORT")
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
            print("\nFACT-CHECK:")
            print("-" * 80)
            print(result["fact_check"])
            print()

    def validation_results_to_string(self, validation_result: Dict) -> str:
        """
        Convert the validation results into a single formatted string.
        
        Args:
            validation_result: Output from validate_prompt()
            
        Returns:
            Formatted string of the entire validation report
        """
        lines = []
        lines.append("="*80)
        lines.append("PRESUMPTION VALIDATION REPORT")
        lines.append("="*80)
        
        lines.append("\nORIGINAL PROMPT:")
        lines.append("-" * 80)
        lines.append(validation_result["original_prompt"])
        
        lines.append(f"\n\nPRESUMPTIONS FOUND: {validation_result['presumptions_found']}")
        lines.append("="*80)
        
        for i, result in enumerate(validation_result["results"], 1):
            lines.append(f"\n{i}. PRESUMPTION:")
            lines.append("-" * 80)
            lines.append(result["presumption"])
            lines.append("\nFACT-CHECK:")
            lines.append("-" * 80)
            lines.append(result["fact_check"])
            lines.append("")
        
        return "\n".join(lines)
    
    def consolidate_results(self, validation_result: Dict) -> str:
        """
        Consolidate all fact-check results into a single summary.
        
        Args:
            validation_result: Output from validate_prompt()
        """

        string_results = self.validation_results_to_string(validation_result)

        consolidation_prompt = """Create a concise version of the following fact-check results. Only include incorrect presumptions and summarize the key facts and evidence for each. Omit any presumptions that were found to be accurate.
        
        {string_results}
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": consolidation_prompt.format(string_results=string_results)}
            ]
        )
        return message.content[0].text
    
    def save_results(self, validation_result: Dict, filename: str):
        with open(filename, "w") as f:
            f.write(self.validation_results_to_string(validation_result))
    
    def save_consolidated(self, consolidated_text: str, filename: str):
        with open(filename, "w") as f:
            f.write(consolidated_text)


def main():
    """
    Example usage of the PresumptionValidator
    """
    # Example prompt
#     example_prompt = """My 70-year-old mom was just diagnosed with lymphoma, but was told by
# her companions that because it is at an advanced stage, no treatment will
# be done. What should we expect?"""
    save_path = "sample_outputs/claude_3.5_haiku"
    os.makedirs(save_path, exist_ok=True)
    for i in range(20):
        example_prompt = datasets.load_from_disk("cancer_myth_dataset")['validation'][i]['question']
        print(f"\n\n=== VALIDATING PROMPT {i+1}/20 ===")
        # Initialize validator (reads API key from .env file)
        try:
            validator = PresumptionValidator(model="claude-3-5-haiku-20241022")
            
            # Run validation
            results = validator.validate_prompt(example_prompt)
            consolidated = validator.consolidate_results(results)
            print("\n\nCONSOLIDATED SUMMARY OF INCORRECT PRESUMPTIONS:")
            print("-" * 80)
            print(consolidated)
            validator.save_consolidated(consolidated, f"{save_path}/consolidated_results_{i}.txt")
            validator.save_results(results, f"{save_path}/full_results_{i}.txt")


            # Print results
            validator.print_results(results)
            
        except ValueError as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()