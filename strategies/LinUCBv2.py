"""
improved linear ucb

features:
- contextual features: bias, recent surv, global surv, usage ratio
- sliding window for A/b (adaptive)
- initial forced exploration
- laplace reward smoothing
- persistent csv history
- compact colored logs
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Dict
from api_client import SphinxAPIClient

HISTORY_FILE = "linear_ucb_history.csv"

# ansi colors
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
        args:
            client: api wrapper
            alpha: exploration strength
            window: history window size
            morty_group_size: morties per send
            exploration_steps: forced exploration count
            history_file: path to csv
        """
        self.client = client
        self.alpha = alpha
        self.window = window
        self.morty_group_size = morty_group_size
        self.exploration_steps = exploration_steps
        self.history_file = history_file

        self.planets = [0, 1, 2]
        # feature dim: bias + recent surv + global surv + usage
        self.d = 4

        # expected history columns
        self.history_cols = [
            "planet",
            "morties_sent",
            "survived",
            "morties_on_jessica",
            "morties_lost",
            "steps_taken"
        ]

        # load history if exists
        if os.path.exists(self.history_file):
            self.history = pd.read_csv(self.history_file)
            for ccol in self.history_cols:
                if ccol not in self.history.columns:
                    raise RuntimeError(f"history missing column '{ccol}'")
            print(c(f"[info] loaded history ({len(self.history)} rows)", CYAN))
        else:
            self.history = pd.DataFrame(columns=self.history_cols)
            print(c("[info] no history found, starting clean", YELLOW))

        # planet matrices (rebuilt each step)
        self.A: Dict[int, np.ndarray] = {}
        self.b: Dict[int, np.ndarray] = {}

    def _save_history(self):
        self.history.to_csv(self.history_file, index=False)
        print(c(f"[saved] {self.history_file} ({len(self.history)} rows)", GRAY))

    def _recent_survival(self, planet: int, n: int) -> float:
        # avg survival over last n trips
        df = self.history[self.history["planet"] == planet].tail(n)
        if len(df) == 0:
            return 0.5
        rates = (df["survived"] / df["morties_sent"]).fillna(0.5)
        return rates.mean()

    def _global_survival(self, planet: int) -> float:
        # avg survival over full history
        df = self.history[self.history["planet"] == planet]
        if len(df) == 0:
            return 0.5
        rates = (df["survived"] / df["morties_sent"]).fillna(0.5)
        return rates.mean()

    def _usage_ratio(self, planet: int) -> float:
        # fraction of all trips sent to planet
        total = len(self.history)
        if total == 0:
            return 1 / len(self.planets)
        return len(self.history[self.history["planet"] == planet]) / total

    def _features(self, planet: int) -> np.ndarray:
        # x = [1, recent, global, usage]
        recent = self._recent_survival(planet, max(1, min(self.window // 4, len(self.history))))
        global_s = self._global_survival(planet)
        usage = self._usage_ratio(planet)
        return np.array([1.0, recent, global_s, usage]).reshape(self.d, 1)

    def _recompute_Ab_from_history(self):
        # rebuild A and b from last window
        for p in self.planets:
            self.A[p] = np.eye(self.d)
            self.b[p] = np.zeros((self.d, 1))

        for p in self.planets:
            recent = self.history[self.history["planet"] == p].tail(self.window)
            for _, row in recent.iterrows():
                x = self._features(p)
                r = (row["survived"] + 1.0) / (row["morties_sent"] + 2.0)  # laplace smooth
                self.A[p] += x @ x.T
                self.b[p] += r * x

    def _linucb_score(self, planet: int) -> float:
        # compute ucb score = mean + alpha * conf
        A = self.A[planet]
        b = self.b[planet]
        try:
            A_inv = np.linalg.inv(A)
        except:
            A_inv = np.linalg.pinv(A)
        theta = A_inv @ b
        x = self._features(planet)
        mean = float((theta.T @ x)[0, 0])
        conf = self.alpha * math.sqrt(float((x.T @ A_inv @ x)[0, 0]))
        return max(0.0, min(mean + conf, 1.5))

    def _append_history_row(self, **kwargs):
        # add row and persist
        self.history = pd.concat([self.history, pd.DataFrame([kwargs])], ignore_index=True)
        self._save_history()

    def execute(self):
        # init state
        status = self.client.get_status()
        morties_remaining = int(status["morties_in_citadel"])
        steps = int(status["steps_taken"])

        print(c("\n=== starting linear ucb v2 ===", MAGENTA))
        print(c(f"window={self.window}, alpha={self.alpha}, group={self.morty_group_size}", GRAY))

        self._recompute_Ab_from_history()

        forced = 0

        while morties_remaining > 0:
            steps += 1

            # forced exploration
            if forced < self.exploration_steps:
                planet = forced % len(self.planets)
                mode = "explore"
                forced += 1
            else:
                # normal ucb
                self._recompute_Ab_from_history()
                scores = {p: self._linucb_score(p) for p in self.planets}
                planet = max(scores, key=scores.get)
                mode = "ucb"

            to_send = min(self.morty_group_size, morties_remaining)
            result = self.client.send_morties(int(planet), int(to_send))

            # parse survival
            survived_raw = result.get("survived", 0)
            if isinstance(survived_raw, bool):
                survived = to_send if survived_raw else 0
            else:
                try:
                    survived = int(survived_raw)
                except:
                    prev_j = int(self.history["morties_on_jessica"].iloc[-1]) if len(self.history) else 0
                    survived = max(0, int(result.get("morties_on_planet_jessica", prev_j)) - prev_j)

            # store history
            self._append_history_row(
                planet=planet,
                morties_sent=to_send,
                survived=survived,
                morties_on_jessica=int(result.get("morties_on_planet_jessica", 0)),
                morties_lost=int(result.get("morties_lost", 0)),
                steps_taken=int(result.get("steps_taken", steps))
            )

            # color logs
            msg = (
                f"[step {result.get('steps_taken', steps)}] {mode:8s} "
                f"p={planet} | sent={to_send} | surv={survived} | "
                f"jess={result.get('morties_on_planet_jessica', '?')} | rem={result.get('morties_in_citadel', '?')}"
            )
            print(c(msg, GREEN if survived > 0 else RED))

            morties_remaining = int(result.get("morties_in_citadel", morties_remaining))

        # final print
        final = self.client.get_status()
        print(c("\n=== done ===", MAGENTA))
        print(c(f"saved: {final['morties_on_planet_jessica']}", GREEN))
        print(c(f"lost : {final['morties_lost']}", RED))
        print(c(f"steps: {final['steps_taken']}", GRAY))
        print(c(f"rate : {final['morties_on_planet_jessica'] / 1000 * 100:.2f}%", YELLOW))


client = SphinxAPIClient()
client.start_episode()
strategy = LinearUCBv2(client, alpha=1.5, window=60, morty_group_size=1, exploration_steps=36)
strategy.execute()
