import numpy as np
from strategy import MortyRescueStrategy
from api_client import SphinxAPIClient


class LinUCBstrategy(MortyRescueStrategy):
    """
    Linear UCB (LinUCB) Strategy with type-safe planet identifiers.
    """

    def __init__(self, client: SphinxAPIClient, alpha: float = 1.0):
        super().__init__(client)
        self.alpha = alpha
        self.A = {}
        self.b = {}
        self.planet_list = []
        self.d = 0

    def _get_context(self, step: int, planet: str) -> np.ndarray:
        """
        Construct context vector with safe planet indexing.
        """
        x = np.zeros(self.d)
        x[0] = step / 1000.0

        planet_str = str(planet)
        if planet_str not in self.planet_list:
            # Handle unseen planet dynamically
            self.planet_list.append(planet_str)
            self.A[planet_str] = np.identity(self.d)
            self.b[planet_str] = np.zeros((self.d, 1))

        planet_index = self.planet_list.index(planet_str)
        x[planet_index + 1] = 1.0
        return x

    def execute_strategy(self, morties_per_trip: int = 3):
        print("\n=== EXECUTING LINUCB STRATEGY ===")

        status = self.client.get_status()
        morties_remaining = int(status["morties_in_citadel"])
        print(f"Starting with {morties_remaining} Morties in Citadel")

        # Use string-safe planet names from exploration data
        self.planet_list = [str(p) for p in self.exploration_data["planet"].unique()]
        self.d = len(self.planet_list) + 1  # +1 for step feature

        # Initialize A and b
        for planet in self.planet_list:
            self.A[planet] = np.identity(self.d)
            self.b[planet] = np.zeros((self.d, 1))

        # Preload exploration data
        for _, row in self.exploration_data.iterrows():
            planet = str(row["planet"])
            survived = int(row["survived"])
            context = self._get_context(0, planet).reshape(-1, 1)
            self.A[planet] += context @ context.T
            self.b[planet] += survived * context

        step = 0
        while morties_remaining > 0:
            ucb_values = {}
            for planet in self.planet_list:
                x = self._get_context(step, planet).reshape(-1, 1)
                A_inv = np.linalg.inv(self.A[planet])
                theta = A_inv @ self.b[planet]
                mean = (theta.T @ x).item()
                confidence = self.alpha * np.sqrt((x.T @ A_inv @ x).item())
                ucb_values[planet] = mean + confidence

            best_planet = max(ucb_values, key=ucb_values.get)
            morties_to_send = int(min(morties_per_trip, morties_remaining))

            result = self.client.send_morties(int(best_planet), int(morties_to_send))

            reward = int(result["survived"]) / morties_to_send
            x = self._get_context(step, best_planet).reshape(-1, 1)
            self.A[best_planet] += x @ x.T
            self.b[best_planet] += reward * x

            morties_remaining = int(result["morties_in_citadel"])
            step += 1

            if step % 50 == 0:
                print(f"\nAfter {step} steps:")
                for planet in self.planet_list:
                    A_inv = np.linalg.inv(self.A[planet])
                    theta = A_inv @ self.b[planet]
                    print(f"  {planet}: predicted success ≈ {theta[0,0]:.2f}")
                print(f"Morties remaining: {morties_remaining}")

        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica'] / 1000) * 100:.2f}%")


if __name__ == "__main__":
    from strategy import run_strategy  
    run_strategy(LinUCBstrategy, explore_trips=60)
