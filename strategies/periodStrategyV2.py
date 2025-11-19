import numpy as np
import math
from api_client import SphinxAPIClient


class PlanetModel:
    def __init__(self, T):
        self.T = T
        self.counts = np.zeros(T, dtype=int)
        self.means = np.full(T, 0.5, dtype=float)
        self.M2 = np.zeros(T, dtype=float)  # for variance tracking

    def update(self, phase, reward):
        c = self.counts[phase]
        m = self.means[phase]

        # welford variance update
        delta = reward - m
        m_new = m + delta / (c + 1)
        self.M2[phase] += delta * (reward - m_new)

        self.means[phase] = m_new
        self.counts[phase] += 1

    def smooth_mean(self, phase):
        """3-point smoothing to remove noise (circular)"""
        T = self.T

        prev = self.means[(phase - 1) % T]
        curr = self.means[phase]
        nxt = self.means[(phase + 1) % T]

        return 0.25 * prev + 0.5 * curr + 0.25 * nxt

    def estimate(self, phase):
        return self.smooth_mean(phase)

    def slope(self, phase):
        """Slope across ±2 phases for stability."""
        T = self.T
        a = self.smooth_mean((phase - 2) % T)
        b = self.smooth_mean((phase + 2) % T)
        return (b - a) / 4

    def std(self, phase):
        c = self.counts[phase]
        if c < 2:
            return 0.3  # assume high uncertainty
        return math.sqrt(self.M2[phase] / (c - 1))

class PhaseStrategy:
    def __init__(self, client):
        self.client = client
        self.T = [10, 20, 200]
        self.models = [PlanetModel(Ti) for Ti in self.T]
        self.step = 0
        self.morties_sent = 0

    def discover_planet(self, p, num_trips):
        """Send exactly `num_trips` trips (1 Morty each) to planet p."""
        print(f"Découverte planète {p} (T={self.T[p]}) avec {num_trips} trips...")
        for _ in range(num_trips):
            if self.morties_sent >= 1000:
                break
            result = self.client.send_morties(p, 1)
            reward = float(result["survived"])
            phase = self.step % self.T[p]
            self.models[p].update(phase, reward)
            self.step += 1
            self.morties_sent += 1
        print(f"  ✔ Planète {p}: {num_trips} trips terminés.")

    def ucb_score(self, p, phase, t):
        model = self.models[p]
        mean = model.estimate(phase)
        cnt = model.counts[phase]
        std = model.std(phase)
        bonus = math.sqrt(math.log(t + 3) / (cnt + 1))
        bonus *= (0.4 + 0.6 * std)
        bonus /= math.sqrt(self.T[p])
        return mean + bonus

    def choose_planet(self):
        t = self.step
        best_score = -1
        best_p = 0
        for p in range(3):
            phase = t % self.T[p]
            score = self.ucb_score(p, phase, t)
            if score > best_score:
                best_score = score
                best_p = p
        return best_p


    def choose_batch(self, est, slope):
        if est > 0.75:
            return 3
        if est > 0.65 and slope > 0:
            return 2
        return 1

    def run(self):
        print("PhaseStrategy — Custom Discovery (10/15/30)")

        # Custom discovery per planet
        self.discover_planet(0, 0)   # Planet 0: 10 trips
        self.discover_planet(1, 0)   # Planet 1: 15 trips
        self.discover_planet(2, 100)   # Planet 2: 30 trips

        # Main exploitation loop
        while self.morties_sent < 1000:
            p = self.choose_planet()
            phase = self.step % self.T[p]
            est = self.models[p].estimate(phase)
            sl = self.models[p].slope(phase)
            batch = self.choose_batch(est, sl)
            batch = min(batch, 1000 - self.morties_sent)

            if batch <= 0:
                break

            result = self.client.send_morties(p, batch)
            reward = float(result["survived"]) / batch
            self.models[p].update(phase, reward)
            self.step += 1
            self.morties_sent += batch

            if self.morties_sent % 100 == 0:
                print(f"[{self.morties_sent}] p={p} est={est:.3f} sl={sl:.3f} batch={batch} surv={result['survived']}")

        print(f"\nFini ! Total = {self.morties_sent} Morties envoyés.")


def main():
    client = SphinxAPIClient()

    print("Nouvelle épisode…")
    client.start_episode()
    print("OK 1000 Mortys prêts !")

    strat = PhaseStrategy(client)
    strat.run()

    status = client.get_status()
    saved = status["morties_on_planet_jessica"]
    lost = status["morties_lost"]

    print("\n" + "="*60)
    print("RESULTAT FINAL")
    print("="*60)
    print(f" Morties sauvés : {saved}/1000")
    print(f" Morties perdus : {lost}/1000")
    print(f" Taux de succès : {saved/1000:.1%}")
    print("="*60)


if __name__ == "__main__":
    main()