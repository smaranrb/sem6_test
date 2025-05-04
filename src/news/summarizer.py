import openai
from typing import Dict, Any
import logging


class NewsSummarizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config["api_keys"]["openai"]
        self.model = config["llm"]["model"]
        self.temperature = config["llm"]["temperature"]
        self.max_tokens = config["llm"]["max_tokens"]
        self.logger = logging.getLogger(__name__)

        # Set OpenAI API key
        openai.api_key = self.api_key

    def summarize(self, text: str) -> str:
        """
        Summarize a news article using OpenAI's LLM.

        Args:
            text: The article text to summarize

        Returns:
            Summarized text
        """
        try:
            # Prepare the prompt
            prompt = f"""Please provide a concise summary of the following financial news article, 
            focusing on key points and market implications:

            {text}

            Summary:"""

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial news summarizer. Provide clear, concise summaries focusing on key market implications.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract and return the summary
            summary = response.choices[0].message.content.strip()
            return summary

        except Exception as e:
            self.logger.error(f"Error summarizing text: {str(e)}")
            return "Error generating summary."

    def analyze_impact(self, text: str) -> Dict[str, Any]:
        """
        Analyze the potential market impact of a news article.

        Args:
            text: The article text to analyze

        Returns:
            Dictionary containing impact analysis
        """
        try:
            # Prepare the prompt
            prompt = f"""Analyze the potential market impact of the following financial news article.
            Provide:
            1. Short-term impact (1-2 days)
            2. Medium-term impact (1-2 weeks)
            3. Key factors affecting the market
            4. Risk level (Low/Medium/High)

            Article:
            {text}

            Analysis:"""

            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial market analyst. Provide detailed market impact analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extract and return the analysis
            analysis = response.choices[0].message.content.strip()

            # Parse the analysis into structured format
            impact_analysis = {
                "analysis": analysis,
                "risk_level": self._extract_risk_level(analysis),
            }

            return impact_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing impact: {str(e)}")
            return {"analysis": "Error generating analysis.", "risk_level": "Unknown"}

    def _extract_risk_level(self, analysis: str) -> str:
        """
        Extract risk level from analysis text.

        Args:
            analysis: The analysis text

        Returns:
            Risk level (Low/Medium/High)
        """
        analysis_lower = analysis.lower()
        if "high risk" in analysis_lower or "high-risk" in analysis_lower:
            return "High"
        elif "medium risk" in analysis_lower or "moderate risk" in analysis_lower:
            return "Medium"
        elif "low risk" in analysis_lower:
            return "Low"
        else:
            return "Unknown"
