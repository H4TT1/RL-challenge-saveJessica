import math
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from api_client import SphinxAPIClient

# ============================================================
#        NUMÉROTATION AUTOMATIQUE DES FICHIERS
# ============================================================

def get_next_run_index():
    """Trouve le prochain numéro de run sous la forme run_XXX.csv."""
    existing = glob.glob("run_*.csv")
    if not existing:
        return 1

    nums = []
    for f in existing:
        base = os.path.basename(f)
        num = base.replace("run_", "").replace(".csv", "")
        if num.isdigit():
            nums.append(int(num))

    if not nums:
        return 1

    return max(nums) + 1


# ============================================================
#     MODELE KALMAN PÉRIODIQUE POUR UNE PLANÈTE
# ============================================================

class PeriodicKalmanPlanet:
    """
    Modèle:
        p(t) ≈ a0 + a_c cos(ω t) + a_s sin(ω t)
    État: θ = [a0, a_c, a_s]^T
    Observation: z_t ≈ H_t θ + bruit,  H_t = [1, cos(ω t), sin(ω t)].
    """

    def __init__(self, T, process_var=0.01, meas_var=0.08):
        self.T = T
        self.omega = 2 * math.pi / T

        # état initial : offset 0.5, peu d'info sur ac/as
        self.theta = pd.Series([0.5, 0.0, 0.0], index=["a0", "ac", "as"])
        self.P = 0.5 * pd.DataFrame(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            columns=["a0", "ac", "as"],
            index=["a0", "ac", "as"],
        )

        # bruit de process (on va le tuner par planète)
        self.Q = process_var * self.P.copy()
        # bruit de mesure (reward binaire mais on veut faire confiance)
        self.R = meas_var

    def _H(self, t):
        """Vecteur d'observation H_t pour le temps t."""
        phi = self.omega * t
        c = math.cos(phi)
        s = math.sin(phi)
        return pd.Series([1.0, c, s], index=["a0", "ac", "as"])

    def predict(self):
        """Kalman prévision: θ ne change pas, P augmente."""
        self.P = self.P + self.Q

    def update(self, t, z):
        """
        Mise à jour avec observation z (reward moyen ∈ [0,1]) au temps t.
        """
        H = self._H(t)                    # (3,)
        h = float(H @ self.theta)         # prédiction
        S = float(H @ self.P @ H) + self.R
        K = (self.P @ H) / S              # gain (3,)

        y = z - h                         # innovation
        self.theta = self.theta + K * y

        # P = (I - K H) P
        I = pd.DataFrame(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            columns=["a0", "ac", "as"],
            index=["a0", "ac", "as"],
        )
        # Correction: K doit être en colonne (3,1) et H en ligne (1,3)
        K_col = K.values.reshape(-1, 1)  # (3,1)
        H_row = H.values.reshape(1, -1)  # (1,3)
        KH_matrix = K_col @ H_row        # (3,3)
        KH = pd.DataFrame(KH_matrix, columns=["a0", "ac", "as"], index=["a0", "ac", "as"])
        self.P = (I - KH) @ self.P

    def prob(self, t):
        """Probabilité prédite au temps t."""
        H = self._H(t)
        p = float(H @ self.theta)
        return max(0.01, min(0.99, p))

    def prob_var(self, t):
        """Variance approximative de la probabilité prédite au temps t."""
        H = self._H(t)
        return float(H @ self.P @ H) + self.R

    def estimate(self, t):
        return self.prob(t)

    def analytic_slope(self, t):
        """
        Dérivée analytique de p(t) = a0 + ac cos(ω t) + as sin(ω t)
        p'(t) = ω(-ac sin(ω t) + as cos(ω t))
        """
        phi = self.omega * t
        c = math.cos(phi)
        s = math.sin(phi)
        ac = float(self.theta["ac"])
        as_ = float(self.theta["as"])
        return self.omega * (-ac * s + as_ * c)


# ============================================================
#              STRATÉGIE KALMAN PÉRIODIQUE + UCB
# ============================================================

class PeriodicKalmanStrategy:
    def __init__(self):
        self.client = SphinxAPIClient()
        self.T = [10, 20, 200]

        # Q plus grand pour T courts, plus petit pour T=200
        # Chaque planète a son propre process_var ET meas_var
        process_vars = [0.03, 0.01, 0.02]
        meas_vars = [0.08, 0.08, 0.08]  # Maintenant différenciable par planète!

        self.planets = [
            PeriodicKalmanPlanet(self.T[i], process_var=process_vars[i], meas_var=meas_vars[i])
            for i in range(3)
        ]

        self.step = 0
        self.morties_sent = 0
        self.log_rows = []

    # --------------------------------------------------------
    def discover_planet(self, p, samples):
        """Phase de découverte: quelques observations sur chaque planète."""
        print(f"Découverte planète {p} ({samples} trips)...")

        for _ in range(samples):
            if self.morties_sent >= 1000:
                break

            planet = self.planets[p]
            t = self.step

            planet.predict()
            batch = 1
            result = self.client.send_morties(p, batch)
            survived = result["survived"]
            reward = survived / batch

            planet.update(t, reward)
            est = planet.estimate(t)

            self.log_rows.append({
                "step": self.step,
                "sent": self.morties_sent,
                "planet": p,
                "estimate": est,
                "slope": 0.0,
                "batch": batch,
                "survived": survived
            })

            self.step += 1
            self.morties_sent += batch

        print(f"  → Fin découverte p={p}, total={self.morties_sent} Morties.")

    # --------------------------------------------------------
    def ucb_score(self, p):
        """Score UCB basé sur mean + incertitude sur la probabilité."""
        planet = self.planets[p]
        t = self.step
        mean = planet.estimate(t)
        var = planet.prob_var(t)

        bonus = 0.6 * math.sqrt(max(var, 1e-6))

        # légère modulation selon période
        if self.T[p] == 10:
            bonus *= 0.7   # un peu moins d'exploration sur la planète très chaotique
        elif self.T[p] == 200:
            bonus *= 1.3   # un peu plus d'exploration sur la planète lente

        return mean + bonus

    # --------------------------------------------------------
    def choose_planet(self):
        best_p = 0
        best_s = -1.0
        for p in range(3):
            s = self.ucb_score(p)
            if s > best_s:
                best_s = s
                best_p = p
        return best_p

    # --------------------------------------------------------
    def choose_batch(self, est, sl):
        """
        Batch en fonction de probabilité & tendance.
        On utilise le slope analytique : si positif, on est en montée.
        """
        if est > 0.75 and sl > 0:
            return 3
        if est > 0.60:
            return 2
        return 1

    # --------------------------------------------------------
    def run(self):
        print("STRATÉGIE KALMAN PÉRIODIQUE (tuning Q/R + slope analytique)")

        # Découverte (un peu plus sur T=200)
        self.discover_planet(0, samples=30)
        self.discover_planet(1, samples=30)
        self.discover_planet(2, samples=30)

        # -------- EXPLOITATION --------
        while self.morties_sent < 1000:
            p = self.choose_planet()
            planet = self.planets[p]

            t = self.step
            planet.predict()

            est_before = planet.estimate(t)
            sl = planet.analytic_slope(t)

            batch = self.choose_batch(est_before, sl)
            batch = min(batch, 1000 - self.morties_sent)
            if batch <= 0:
                break

            result = self.client.send_morties(p, batch)
            survived = result["survived"]
            reward = survived / batch

            planet.update(t, reward)
            est_after = planet.estimate(t)
            sl_after = planet.analytic_slope(t)

            self.log_rows.append({
                "step": self.step,
                "sent": self.morties_sent,
                "planet": p,
                "estimate": est_after,
                "slope": sl_after,
                "batch": batch,
                "survived": survived
            })

            self.step += 1
            self.morties_sent += batch

            if self.morties_sent % 100 == 0 or self.morties_sent >= 1000:
                print(f"[{self.morties_sent} sent] p={p} est={est_after:.2f} "
                      f"sl={sl_after:.2f} batch={batch} surv={survived}")

        print(f"Run terminé. {self.morties_sent} Morties envoyés.")

        self.save_logs_and_plot()

    # --------------------------------------------------------
    def save_logs_and_plot(self):
        run_id = get_next_run_index()
        csv_name = f"run_{run_id:03d}.csv"
        png_name = f"run_{run_id:03d}.png"

        df = pd.DataFrame(self.log_rows)
        df.to_csv(csv_name, index=False)
        print(f"CSV sauvegardé : {csv_name}")

        plt.figure(figsize=(12, 6))
        plt.plot(df["step"], df["estimate"], label="estimate")
        plt.plot(df["step"], df["slope"], label="slope")
        plt.scatter(df["step"], df["survived"], alpha=0.3, label="survived")
        plt.title(f"Run {run_id:03d} (Kalman périodique v2)")
        plt.xlabel("Step")
        plt.ylabel("Values")
        plt.legend()
        plt.grid()
        plt.savefig(png_name)
        plt.close()
        print(f"Plot sauvegardé : {png_name}")


# ============================================================
#                           MAIN
# ============================================================

def main():
    client = SphinxAPIClient()
    print("Nouvelle épisode…")
    client.start_episode()
    print("OK 1000 Mortys prêts !")

    strat = PeriodicKalmanStrategy()
    strat.run()

    status = client.get_status()
    saved = status['morties_on_planet_jessica']
    lost = status['morties_lost']
    success_rate = saved / 1000

    print("\n" + "=" * 60)
    print("RÉSULTAT FINAL:")
    print("=" * 60)
    print(f" Morties sauvés: {saved}/1000")
    print(f" Morties perdus: {lost}/1000")
    print(f" Taux de succès: {success_rate:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
