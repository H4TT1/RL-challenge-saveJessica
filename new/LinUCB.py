import numpy as np
import pandas as pd
from api_client import SphinxAPIClient
from typing import Dict

class LinearUCB:
    """
    Linear UCB strategy for Morty Express.
    Each planet has its own linear model with:
        A_p : (d x d) design matrix
        b_p : (d x 1) response vector
    """

    def __init__(self, client: SphinxAPIClient, alpha=1.5, window=30, morty_group_size=3):
        self.client = client
        self.alpha = alpha
        self.window = window
        self.morty_group_size = morty_group_size

        self.planets = [0, 1, 2]

        # d = number of features in x
        self.d = 3

        # LinUCB matrices
        self.A = {p: np.identity(self.d) for p in self.planets}
        self.b = {p: np.zeros((self.d, 1)) for p in self.planets}

        # Store history
        self.history = pd.DataFrame(columns=["planet", "survived", "morties_sent"])

    def _decay(self, planet, decay=0.995):
        self.A[planet] *= decay
        self.b[planet] *= decay


    def _update_history(self, planet, survived, morties_sent):
        """Keep track of the last trips."""
        self.history = pd.concat(
            [
                self.history,
                pd.DataFrame([
                    {"planet": planet, "survived": int(survived), "morties_sent": morties_sent}
                ])
            ],
            ignore_index=True
        )

    def _recent_survival(self, planet):
        """Short-term survival over sliding window."""
        df = self.history[self.history["planet"] == planet].tail(self.window)
        if len(df) == 0: return 0.5
        return df["survived"].mean()

    def _global_survival(self, planet):
        """All-time survival rate."""
        df = self.history[self.history["planet"] == planet]
        if len(df) == 0: return 0.5
        return df["survived"].mean()

    def _features(self, planet):
        """Feature vector x_p ∈ ℝ³."""
        return np.array([
            1.0,
            self._recent_survival(planet),
            self._global_survival(planet),
        ])

    def _ucb_score(self, planet):
        """
        Compute LinUCB score:
            θ_hat = A⁻¹ b
            p = θ_hatᵀ x + α sqrt(xᵀ A⁻¹ x)
        """
        A_inv = np.linalg.inv(self.A[planet])
        theta_hat = A_inv @ self.b[planet]

        x = self._features(planet).reshape(-1, 1)

        mean_reward = float((theta_hat.T @ x)[0][0])
        confidence = self.alpha * float(np.sqrt((x.T @ A_inv @ x)[0][0]))

        return mean_reward + confidence

    def _update(self, planet, reward):
        """
        Update A_p and b_p based on outcome.
        Reward = survived (0 or 1)
        """
        x = self._features(planet).reshape(-1, 1)
        self.A[planet] += x @ x.T
        self.b[planet] += reward * x

    def execute(self):
        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]

        print("\n=== Starting Linear UCB ===")

        while morties_remaining > 0:

            # Compute score for each planet
            scores = {p: self._ucb_score(p) for p in self.planets}
            planet = max(scores, key=scores.get)

            to_send = min(self.morty_group_size, morties_remaining)

            # Send Morties
            result = self.client.send_morties(planet, to_send)
            survived = result["survived"]

            # Reward = survival rate
            reward = survived / to_send

            # Update LinUCB model
            self._decay(planet)
            self._update(planet, reward)
            self._update_history(planet, survived, to_send)

            morties_remaining = result["morties_in_citadel"]

            if result["steps_taken"] % 50 == 0:
                print(f"[Step {result['steps_taken']}] Planet {planet} — Reward={reward:.2f}")

        final = self.client.get_status()

        print("\n=== LINEAR UCB DONE ===")
        print(f"Morties Saved: {final['morties_on_planet_jessica']}")
        print(f"Morties Lost:  {final['morties_lost']}")
        print(f"Total Steps:   {final['steps_taken']}")
        print(f"Success Rate:  {final['morties_on_planet_jessica'] / 1000 * 100:.2f}%")


if __name__ == "__main__":
    client = SphinxAPIClient()
    client.start_episode()  # reset game
    strategy = LinearUCB(client, window=100, morty_group_size=2)
    strategy.execute()
