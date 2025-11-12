import math
import pandas as pd
from collections import defaultdict
from api_client import SphinxAPIClient
from data_collector import DataCollector
from strategy import MortyRescueStrategy 


class UCBStrategy(MortyRescueStrategy):
    """
    Upper Confidence Bound (UCB) Strategy for Morty Express Challenge.
    Balances exploration and exploitation dynamically.
    """
    
    def __init__(self, client: SphinxAPIClient, exploration_coef: float = 2):
        super().__init__(client)
        self.exploration_coef = exploration_coef
        self.planet_stats = defaultdict(lambda: {"successes": 0, "trips": 0})
    
    def execute_strategy(self, morties_per_trip: int = 3):
        """
        Execute UCB-based strategy to decide which planet to send Morties to.
        
        Args:
            morties_per_trip: Number of Morties per trip (1-3)
        """
        print("\n=== EXECUTING UCB STRATEGY ===")
        
        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        print(f"Starting with {morties_remaining} Morties in Citadel")

        # Initialize planet list from exploration data
        planets = self.exploration_data["planet"].unique()
        
        # Initialize stats from exploration data
        for planet in planets:
            planet_data = self.exploration_data[self.exploration_data["planet"] == planet]
            successes = planet_data["survived"].sum()
            trips = len(planet_data)
            self.planet_stats[planet]["successes"] = successes
            self.planet_stats[planet]["trips"] = trips
        
        total_trips = sum(stats["trips"] for stats in self.planet_stats.values())
        
        # UCB decision loop
        while morties_remaining > 0:
            # Compute UCB for each planet
            ucb_values = {}
            for planet, stats in self.planet_stats.items():
                n_i = stats["trips"]
                if n_i == 0:
                    ucb_values[planet] = float("inf")  # force exploration
                else:
                    mean_success = stats["successes"] / n_i
                    confidence = self.exploration_coef * math.sqrt(math.log(total_trips + 1) / n_i)
                    ucb_values[planet] = mean_success + confidence
            
            # Pick best planet
            best_planet = max(ucb_values, key=ucb_values.get)
            
            # Send Morties
            morties_to_send = min(morties_per_trip, morties_remaining)
            result = self.client.send_morties(int(best_planet), int(morties_to_send))

            
            # Update stats
            survived = result["survived"]
            self.planet_stats[best_planet]["successes"] += survived
            self.planet_stats[best_planet]["trips"] += 1
            
            total_trips += 1
            morties_remaining = result["morties_in_citadel"]
            
            # Logging progress
            if total_trips % 50 == 0:
                avg_success = {
                    p: (s["successes"] / s["trips"]) if s["trips"] > 0 else 0
                    for p, s in self.planet_stats.items()
                }
                print(f"\nAfter {total_trips} trips:")
                for planet, rate in avg_success.items():
                    print(f"  {planet}: {rate*100:.2f}% success rate")
                print(f"Morties remaining: {morties_remaining}")
        
        # Final summary
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica'] / 1000) * 100:.2f}%")


# Example usage
if __name__ == "__main__":
    from strategy import run_strategy  
    run_strategy(UCBStrategy, explore_trips=60)
