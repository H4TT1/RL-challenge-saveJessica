import math
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from api_client import SphinxAPIClient


def get_next_run_index():
    existing = glob.glob("run_*.csv")
    if not existing:
        return 1

    nums = []
    for f in existing:
        base = os.path.basename(f)
        num = base.replace("run_", "").replace(".csv", "")
        if num.isdigit():
            nums.append(int(num))

    return max(nums) + 1 if nums else 1


class PeriodicKalmanPlanet:
    """
    p(t) = a0 + ac cos(ωt) + as sin(ωt)
    θ = [a0, ac, as]
    """

    def __init__(self, T, process_var=0.01, meas_var=0.08):
        self.T = T
        self.omega = 2 * math.pi / T

        self.theta = pd.Series([0.5, 0.0, 0.0], index=["a0", "ac", "as"])
        self.P = 0.5 * pd.DataFrame(
            [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]],
            index=["a0", "ac", "as"],
            columns=["a0", "ac", "as"]
        )

        self.Q = process_var * self.P.copy()
        self.R = meas_var

    def _H(self, t):
        phi = self.omega * t
        return pd.Series([1.0, math.cos(phi), math.sin(phi)],
                         index=["a0", "ac", "as"])

    def predict(self):
        self.P = self.P + self.Q

    def update(self, t, z):
        H = self._H(t)
        h = float(H @ self.theta)
        S = float(H @ self.P @ H) + self.R
        K = (self.P @ H) / S

        y = z - h
        self.theta = self.theta + K * y

        I = pd.DataFrame(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            index=["a0", "ac", "as"],
            columns=["a0", "ac", "as"]
        )

        K_col = K.values.reshape(3, 1)
        H_row = H.values.reshape(1, 3)
        KH = pd.DataFrame(K_col @ H_row,
                          index=["a0", "ac", "as"],
                          columns=["a0", "ac", "as"])

        self.P = (I - KH) @ self.P

    def estimate(self, t):
        H = self._H(t)
        v = float(H @ self.theta)
        return max(0.01, min(0.99, v))

    def prob_var(self, t):
        H = self._H(t)
        return float(H @ self.P @ H) + self.R

    def analytic_slope(self, t):
        phi = self.omega * t
        ac = float(self.theta["ac"])
        as_ = float(self.theta["as"])
        return self.omega * (-ac * math.sin(phi) + as_ * math.cos(phi))


class PeriodicKalmanStrategy:
    def __init__(self):
        self.client = SphinxAPIClient()

        self.T = [10, 20, 200]
        process_vars = [0.03, 0.01, 0.02]
        meas_vars = [0.08, 0.08, 0.08]

        self.planets = [
            PeriodicKalmanPlanet(self.T[i], process_vars[i], meas_vars[i])
            for i in range(3)
        ]

        self.step = 0
        self.total_sent = 0
        self.log_rows = []

    def discover_planet(self, p, samples):
        for _ in range(samples):
            if self.total_sent >= 1000:
                break

            planet = self.planets[p]
            t = self.step

            planet.predict()

            result = self.client.send_morties(p, 1)
            survived = result["survived"]

            planet.update(t, survived)

            self.log_rows.append({
                "step": self.step,
                "planet": p,
                "estimate": planet.estimate(t),
                "slope": 0.0,
                "batch": 1,
                "survived": survived
            })

            self.step += 1
            self.total_sent += 1

    def ucb_score(self, p):
        planet = self.planets[p]
        t = self.step

        mean = planet.estimate(t)
        var = planet.prob_var(t)
        bonus = 0.6 * math.sqrt(max(var, 1e-6))

        if self.T[p] == 10:
            bonus *= 0.7
        elif self.T[p] == 200:
            bonus *= 1.3

        return mean + bonus

    def choose_planet(self):
        return max(range(3), key=lambda p: self.ucb_score(p))

    def choose_batch(self, est, slope):
        if est > 0.75 and slope > 0:
            return 3
        if est > 0.60:
            return 2
        return 1

    def run(self):
        self.discover_planet(0, 30)
        self.discover_planet(1, 30)
        self.discover_planet(2, 30)

        while self.total_sent < 1000:
            p = self.choose_planet()
            planet = self.planets[p]
            t = self.step

            planet.predict()

            est = planet.estimate(t)
            sl = planet.analytic_slope(t)

            batch = self.choose_batch(est, sl)
            batch = min(batch, 1000 - self.total_sent)

            result = self.client.send_morties(p, batch)
            survived = result["survived"]
            reward = survived / batch

            planet.update(t, reward)

            self.log_rows.append({
                "step": self.step,
                "planet": p,
                "estimate": planet.estimate(t),
                "slope": planet.analytic_slope(t),
                "batch": batch,
                "survived": survived
            })

            self.step += 1
            self.total_sent += batch

        self.save_logs()

    def save_logs(self):
        run_id = get_next_run_index()
        csv_name = f"run_{run_id:03d}.csv"
        png_name = f"run_{run_id:03d}.png"

        df = pd.DataFrame(self.log_rows)
        df.to_csv(csv_name, index=False)

        plt.figure(figsize=(12, 6))
        plt.plot(df["step"], df["estimate"], label="estimate")
        plt.plot(df["step"], df["slope"], label="slope")
        plt.scatter(df["step"], df["survived"], alpha=0.3, label="survived")
        plt.legend()
        plt.grid()
        plt.title(f"Run {run_id:03d}")
        plt.savefig(png_name)
        plt.close()

        print(f"Sauvegardé: {csv_name}, {png_name}")


def main():
    client = SphinxAPIClient()
    print("Nouvel épisode…")
    client.start_episode()
    print("1000 Mortys prêts.")

    strat = PeriodicKalmanStrategy()
    strat.run()

    status = client.get_status()
    saved = status["morties_on_planet_jessica"]
    lost = status["morties_lost"]

    print(f"Résultat final: {saved}/1000 sauvés, {lost} perdus")


if __name__ == "__main__":
    main()
