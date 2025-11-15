import math
from collections import defaultdict, deque
from api_client import SphinxAPIClient
from data_collector import DataCollector
from strategy import MortyRescueStrategy  # adjust import path if needed


class SlidingWindowUCBStrategy(MortyRescueStrategy):
    """
    Sliding-Window UCB Strategy.
    Adaptively balances exploration and exploitation for non-stationary rewards.
    """

    def __init__(self, client: SphinxAPIClient, exploration_coef: float = 2.0, window_size: int = 100):
        super().__init__(client)
        self.exploration_coef = exploration_coef
        self.window_size = window_size
        # For each planet, maintain a sliding window of 0/1 (failure/success)
        self.planet_history = defaultdict(lambda: deque(maxlen=self.window_size))

    def execute_strategy(self, morties_per_trip: int = 3):
        print("\n=== EXECUTING SLIDING-WINDOW UCB STRATEGY ===")

        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        print(f"Starting with {morties_remaining} Morties in Citadel")

        # Get planet list from exploration data
        planets = list(self.exploration_data["planet"].unique())

        # Initialize from exploration phase
        for planet in planets:
            planet_data = self.exploration_data[self.exploration_data["planet"] == planet]
            for survived in planet_data["survived"].astype(int):
                self.planet_history[planet].append(int(survived))

        total_trips = sum(len(v) for v in self.planet_history.values())

        while morties_remaining > 0:
            # Compute UCB for each planet
            ucb_values = {}
            for planet, history in self.planet_history.items():
                n_i = len(history)
                if n_i == 0:
                    ucb_values[planet] = float("inf")  # Force initial exploration
                else:
                    mean_success = sum(history) / n_i
                    confidence = self.exploration_coef * math.sqrt(math.log(total_trips + 1) / n_i)
                    ucb_values[planet] = mean_success + confidence

            # Pick planet with highest UCB
            best_planet = max(ucb_values, key=ucb_values.get)
            morties_to_send = min(morties_per_trip, morties_remaining)

            # Ensure JSON-safe types
            best_planet = str(best_planet)
            morties_to_send = int(morties_to_send)

            # Send Morties and record result
            result = self.client.send_morties(int(best_planet), int(morties_to_send))

            survived = int(result["survived"])
            self.planet_history[best_planet].append(survived)

            total_trips += 1
            morties_remaining = result["morties_in_citadel"]

            # Log progress every 50 trips
            if total_trips % 50 == 0:
                print(f"\nAfter {total_trips} trips:")
                for planet, history in self.planet_history.items():
                    if len(history) > 0:
                        avg_rate = sum(history) / len(history)
                        print(f"  {planet}: {avg_rate*100:.2f}% recent success")
                print(f"Morties remaining: {morties_remaining}")

        # Final results
        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica'] / 1000) * 100:.2f}%")


# Example usage
if __name__ == "__main__":
    from strategy import run_strategy  # adjust import if needed
    run_strategy(SlidingWindowUCBStrategy, explore_trips=3)
