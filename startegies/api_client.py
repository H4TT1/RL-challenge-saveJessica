"""
API Client for Sphinx Morty Express Challenge
Base URL: https://challenge.sphinxhq.com
"""

import requests
import os
from typing import Dict, Optional, Any
from dotenv import load_dotenv


class SphinxAPIClient:
    """Client for interacting with the Sphinx Morty Express Challenge API."""
    
    BASE_URL = "https://challenge.sphinxhq.com"
    PLANET_NAMES = {
        0: '"On a Cob" Planet',
        1: "Cronenberg World",
        2: "The Purge Planet"
    }
    
    def __init__(self, api_token: Optional[str] = None):
        """
        Initialize the API client.
        
        Args:
            api_token: API token for authentication. If not provided, 
                      will try to load from SPHINX_API_TOKEN environment variable.
        """
        load_dotenv() # Load environment variables from .env
        self.api_token = api_token or os.getenv("SPHINX_API_TOKEN")
        
        if not self.api_token:
            raise ValueError(
                "API token is required. Please set SPHINX_API_TOKEN environment variable."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
    def _post(self, endpoint: str, json_data: Optional[Dict] = None) -> Dict:
        """Internal POST method with error handling."""
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.post(url, json=json_data, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def _get(self, endpoint: str) -> Dict:
        """Internal GET method with error handling."""
        url = f"{self.BASE_URL}{endpoint}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def request_token(self, name: str, email: str) -> Dict:
        """
        Request an API token (one-time setup). Token will be sent to the email.
        """
        return self._post(
            "/api/auth/request-token/",
            json_data={"name": name, "email": email}
        )
    
    def start_episode(self) -> Dict:
        """Start a new episode, resetting game state."""
        print("Starting new episode...")
        return self._post("/api/mortys/start/")
    
    def send_morties(self, planet: int, morty_count: int) -> Dict:
        """
        Send Morties through a portal.
        
        Args:
            planet: Planet index (0, 1, or 2)
            morty_count: Number of Morties to send (1-3)
        """
        if planet not in self.PLANET_NAMES:
            raise ValueError("Planet must be 0, 1, or 2")
        if morty_count not in [1, 2, 3]:
            raise ValueError("Morty count must be 1, 2, or 3")
        
        return self._post(
            "/api/mortys/portal/",
            json_data={"planet": planet, "morty_count": morty_count}
        )
    
    def get_status(self) -> Dict:
        """Get current episode status."""
        return self._get("/api/mortys/status/")
    
    def get_planet_name(self, planet_index: int) -> str:
        """Get the name of a planet by its index."""
        return self.PLANET_NAMES.get(planet_index, "Unknown Planet")


if __name__ == "__main__":
    try:
        client = SphinxAPIClient()
        print("API Client initialized successfully!")
        status = client.get_status()
        print(f"Current Morties in Citadel: {status['morties_in_citadel']}")
    except Exception as e:
        print(f"Error: {e}")