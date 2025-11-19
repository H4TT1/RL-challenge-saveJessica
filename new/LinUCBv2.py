"""
linear_ucb_v2.py

Improved Linear UCB for the Morty Express Challenge.

Features:
- Contextual (features per planet): bias, recent survival, global survival, relative usage
- Sliding-window re-computation of A and b per planet from recent history (adaptive)
- Explicit exploration phase (force initial visits)
- Reward smoothing (Laplace-style) to reduce noise
- Persistent history saved to CSV so future episodes can reuse data
- Step-by-step colored logging including Morties on Jessica
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict
from api_client import SphinxAPIClient

HISTORY_FILE = "linear_ucb_history.csv"

# ANSI color helpers
def c(text: str, color_code: int) -> str:
    return f"\033[{color_code}m{text}\033[0m"

GREEN = 92
RED = 91
YELLOW = 93
CYAN = 96
MAGENTA = 95
GRAY = 90


class LinearUCBv2:
    def __init__(
        self,
        client: SphinxAPIClient,
        alpha: float = 1.5,
        window: int = 100,
        morty_group_size: int = 3,
        exploration_steps: int = 30,
        history_file: str = HISTORY_FILE,
    ):
        """
        Args:
            client: SphinxAPIClient instance
            alpha: exploration multiplier for confidence term
            window: sliding window size (number of past trips to consider per planet)
            morty_group_size: number of Morties per trip (1-3)
            exploration_steps: number of forced exploratory trips at start
            history_file: path to CSV where history is persisted
        """
        self.client = client
        self.alpha = alpha
        self.window = window
        self.morty_group_size = morty_group_size
        self.exploration_steps = exploration_steps
        self.history_file = history_file

        self.planets = [0, 1, 2]
        # feature dimension: bias + recent_surv + global_surv + usage_ratio
        self.d = 4

        # history DataFrame columns:
        self.history_cols = [
            "planet",
            "morties_sent",
            "survived",                   # integer count survived in that trip (0..morties_sent)
            "morties_on_jessica",         # cumulative after trip or absolute (API gives absolute)
            "morties_lost",
            "steps_taken"
        ]

        # Load history if exists
        if os.path.exists(self.history_file):
            self.history = pd.read_csv(self.history_file)
            # ensure columns present
            for ccol in self.history_cols:
                if ccol not in self.history.columns:
                    raise RuntimeError(f"History file missing expected column '{ccol}'")
            print(c(f"[INFO] Loaded history ({len(self.history)} rows) from {self.history_file}", CYAN))
        else:
            self.history = pd.DataFrame(columns=self.history_cols)
            print(c("[INFO] No previous history found — starting fresh.", YELLOW))

        # matrices will be computed from sliding-window history when needed
        # we keep them in dicts for convenience
        self.A: Dict[int, np.ndarray] = {}
        self.b: Dict[int, np.ndarray] = {}

    # -------------------------
    # Persistence helpers
    # -------------------------
    def _save_history(self):
        self.history.to_csv(self.history_file, index=False)
        print(c(f"[HISTORY SAVED] {self.history_file} ({len(self.history)} rows)", GRAY))

    # -------------------------
    # Feature engineering
    # -------------------------
    def _recent_survival(self, planet: int, n: int) -> float:
        df = self.history[self.history["planet"] == planet].tail(n)
        if len(df) == 0:
            return 0.5
        # per-trip survival proportion: survived / morties_sent averaged
        rates = (df["survived"] / df["morties_sent"]).fillna(0.5)
        return rates.mean()

    def _global_survival(self, planet: int) -> float:
        df = self.history[self.history["planet"] == planet]
        if len(df) == 0:
            return 0.5
        rates = (df["survived"] / df["morties_sent"]).fillna(0.5)
        return rates.mean()

    def _usage_ratio(self, planet: int) -> float:
        # fraction of total trips sent to this planet (recently)
        total = len(self.history)
        if total == 0:
            return 1.0 / len(self.planets)
        planet_count = len(self.history[self.history["planet"] == planet])
        return planet_count / total

    def _features(self, planet: int) -> np.ndarray:
        # x = [1, recent_surv, global_surv, usage_ratio]
        recent = self._recent_survival(planet, max(1, min(self.window // 4, len(self.history))))  # smaller window for "recent" piece
        global_s = self._global_survival(planet)
        usage = self._usage_ratio(planet)
        x = np.array([1.0, recent, global_s, usage], dtype=float).reshape(self.d, 1)
        return x

    # -------------------------
    # Build A and b from sliding-window history
    # -------------------------
    def _recompute_Ab_from_history(self):
        """
        For each planet p: compute
            A_p = I_d + sum_{(x,r) in recent_history_p} x x^T
            b_p = sum_{(x,r)} r x
        where r is a smoothed reward in [0,1] (survived / morties_sent with Laplace smoothing).
        """
        # initialize
        for p in self.planets:
            self.A[p] = np.eye(self.d)
            self.b[p] = np.zeros((self.d, 1))

        # We'll iterate over history and add only last `window` entries per planet.
        for p in self.planets:
            recent = self.history[self.history["planet"] == p].tail(self.window)
            for _, row in recent.iterrows():
                # reconstruct features at that time using current modifiers (approximate)
                x = self._features(p)  # approximate features (uses global history including that row)
                # reward smoothing: (survived + 1) / (morties_sent + 2)
                surv = float(row["survived"])
                sent = float(row["morties_sent"])
                r = (surv + 1.0) / (sent + 2.0)
                self.A[p] += x @ x.T
                self.b[p] += r * x

    # -------------------------
    # LinUCB scoring
    # -------------------------
    def _linucb_score(self, planet: int) -> float:
        # if history is empty for planet, score should encourage exploration
        # but we will still compute with A and b (initialized to I and zero)
        A = self.A[planet]
        b = self.b[planet]
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            A_inv = np.linalg.pinv(A)
        theta = A_inv @ b
        x = self._features(planet)
        mean_reward = float((theta.T @ x)[0, 0])
        conf = self.alpha * math.sqrt(float((x.T @ A_inv @ x)[0, 0]))
        score = mean_reward + conf
        # clip score to plausible [0, 1.5] range to avoid blow-ups
        return max(0.0, min(score, 1.5))

    # -------------------------
    # Update after observing a trip result
    # -------------------------
    def _append_history_row(self, planet: int, morties_sent: int, survived: int, morties_on_jessica: int, morties_lost: int, steps_taken: int):
        new_row = {
            "planet": int(planet),
            "morties_sent": int(morties_sent),
            "survived": int(survived),
            "morties_on_jessica": int(morties_on_jessica),
            "morties_lost": int(morties_lost),
            "steps_taken": int(steps_taken),
        }
        self.history = pd.concat([self.history, pd.DataFrame([new_row])], ignore_index=True)
        # persist
        self._save_history()

    # -------------------------
    # Main execution
    # -------------------------
    def execute(self):
        # get initial status
        status = self.client.get_status()
        morties_remaining = int(status["morties_in_citadel"])
        steps = int(status["steps_taken"])

        print(c("\n=== Starting LinearUCB v2 ===", MAGENTA))
        print(c(f"Window={self.window}, alpha={self.alpha}, group={self.morty_group_size}, exploration_steps={self.exploration_steps}", GRAY))

        # initial recompute
        self._recompute_Ab_from_history()

        forced_count = 0
        total_sent = 0

        while morties_remaining > 0:
            steps += 1

            # if we are in forced exploration period, pick planets round-robin (or random)
            if forced_count < self.exploration_steps:
                planet = forced_count % len(self.planets)
                forced_mode = True
                forced_count += 1
                mode_str = "EXPLORATION"
            else:
                # recompute A and b from the latest history window
                self._recompute_Ab_from_history()
                # compute LinUCB scores and pick best planet
                scores = {p: self._linucb_score(p) for p in self.planets}
                planet = max(scores, key=scores.get)
                forced_mode = False
                mode_str = "LINUCB"

            to_send = min(self.morty_group_size, morties_remaining)
            result = self.client.send_morties(int(planet), int(to_send))

            # environment returns survived as boolean or integer? Most earlier code used result["survived"]
            # If survived is boolean or true when entire group survived, handle both cases:
            survived_raw = result.get("survived", 0)
            # the API previously returned 'survived': true/false for group survival; but also 'morties_sent' etc.
            # We'll compute 'survived_count' as:
            if isinstance(survived_raw, bool):
                # survived flag indicates entire group survived -> count = to_send if True else 0
                survived_count = to_send if survived_raw else 0
            else:
                # assume integer count where possible
                try:
                    survived_count = int(survived_raw)
                except Exception:
                    # fallback: if the API doesn't give count, infer from morties_on_planet_jessica change
                    prev_on_jessica = int(self.history["morties_on_jessica"].iloc[-1]) if len(self.history) > 0 else 0
                    survived_count = max(0, int(result.get("morties_on_planet_jessica", prev_on_jessica)) - prev_on_jessica)

            # Smooth reward for update purposes:
            reward = (survived_count + 1.0) / (to_send + 2.0)  # Laplace smoothing

            # Append to history and persist
            self._append_history_row(
                planet=planet,
                morties_sent=to_send,
                survived=survived_count,
                morties_on_jessica=result.get("morties_on_planet_jessica", 0),
                morties_lost=result.get("morties_lost", 0),
                steps_taken=result.get("steps_taken", steps)
            )

            # Quick logging per step
            step_info = (
                f"[Step {result.get('steps_taken', steps)}] "
                f"{mode_str:11s} Planet {planet} | Sent={to_send} | Survived={survived_count} | "
                f"OnJessica={result.get('morties_on_planet_jessica', '?')} | Remaining={result.get('morties_in_citadel', '?')}"
            )
            # color important numbers
            if survived_count > 0:
                print(c(step_info, GREEN))
            else:
                print(c(step_info, RED))

            morties_remaining = int(result.get("morties_in_citadel", morties_remaining))
            total_sent += to_send

            # occasionally print internal model diagnostics
            if steps % 50 == 0:
                # compute current scores for debug
                self._recompute_Ab_from_history()
                scores = {p: self._linucb_score(p) for p in self.planets}
                debug = " | ".join([f"P{p}: score={scores[p]:.3f}" for p in self.planets])
                print(c(f"[DEBUG] {debug}", CYAN))

        # Final status
        final = self.client.get_status()
        print(c("\n=== LINEAR UCB v2 COMPLETE ===", MAGENTA))
        print(c(f"Morties Saved: {final['morties_on_planet_jessica']}", GREEN))
        print(c(f"Morties Lost : {final['morties_lost']}", RED))
        print(c(f"Total Steps  : {final['steps_taken']}", GRAY))
        print(c(f"Success Rate : {final['morties_on_planet_jessica'] / 1000 * 100:.2f}%", YELLOW))


client = SphinxAPIClient()
client.start_episode()   # reset episode
strategy = LinearUCBv2(client, alpha=1.5, window=60, morty_group_size=1, exploration_steps=36)
strategy.execute()