import pandas as pd
import numpy as np
from api_client import SphinxAPIClient

class UCBStrategy:
    """Upper Confidence Bound (UCB1) strategy for dynamic Morty allocation."""

    def __init__(self, client: SphinxAPIClient, morty_group_size: int = 3):
        self.client = client
        self.morty_group_size = morty_group_size
        self.planets = [0, 1, 2]

        # Track history and counts
        self.history = pd.DataFrame(columns=["planet", "survived", "morties_sent"])
        self.n_i = {p: 0 for p in self.planets}  # Trips per planet
        self.success_i = {p: 0 for p in self.planets}  # Successful trips per planet
        self.total_trips = 0

    def _update_history(self, planet: int, survived: bool, morties_sent: int):
        self.history = pd.concat([
            self.history,
            pd.DataFrame([{"planet": planet, "survived": int(survived), "morties_sent": morties_sent}])
        ], ignore_index=True)
        self.n_i[planet] += 1
        self.success_i[planet] += int(survived)
        self.total_trips += 1

    def _calculate_ucb(self):
        """Calculate UCB value for each planet."""
        ucb_values = {}
        for p in self.planets:
            if self.n_i[p] == 0:
                # Ensure each planet is tried at least once
                ucb_values[p] = float('inf')
            else:
                avg_success = self.success_i[p] / self.n_i[p]
                confidence = np.sqrt(2 * np.log(self.total_trips) / self.n_i[p])
                ucb_values[p] = avg_success + confidence
        return ucb_values

    def execute(self):
        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]

        print(f"\nStarting UCB Strategy (Group Size={self.morty_group_size})")

        while morties_remaining > 0:
            # 1️⃣ Compute UCB for all planets
            ucb_values = self._calculate_ucb()
            best_planet = max(ucb_values, key=ucb_values.get)
            best_planet_name = self.client.get_planet_name(best_planet)

            # 2️⃣ Determine Morties to send
            to_send = min(self.morty_group_size, morties_remaining)

            # 3️⃣ Send Morties
            result = self.client.send_morties(best_planet, to_send)

            # 4️⃣ Update history and counts
            self._update_history(best_planet, result["survived"], result["morties_sent"])
            morties_remaining = result["morties_in_citadel"]

            # 5️⃣ Print step
            print(f"[Step {result['steps_taken']}] Planet: {best_planet_name}, "
                  f"Sent: {to_send}, Survived: {result['survived']}, "
                  f"Morties on Jessica: {result['morties_on_planet_jessica']}, "
                  f"Morties Remaining: {morties_remaining}")

        # ✅ Final results
        final = self.client.get_status()
        print("\n=== UCB STRATEGY COMPLETE ===")
        print(f"Morties Saved on Jessica: {final['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final['morties_lost']}")
        print(f"Total Steps: {final['steps_taken']}")
        success_rate = (final['morties_on_planet_jessica'] / 1000) * 100
        print(f"SUCCESS RATE: {success_rate:.2f}%")
        print("=" * 50)


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    client = SphinxAPIClient()
    client.start_episode()  # Reset the game

    strategy = UCBStrategy(client, morty_group_size=3)
    strategy.execute()
