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
    Observation: z_t ≈ H_t θ + bruit,  H_t = [1, cos(ω t), sin(ω t)]
    """
    def __init__(self, T, process_var=0.001, meas_var=0.15):
        self.T = T
        self.omega = 2 * math.pi / T

        # état initial : offset 0.5, amplitude faible
        self.theta = pd.Series([0.5, 0.0, 0.0], index=["a0", "ac", "as"])
        self.P = 0.5 * pd.DataFrame(
            [[1.0, 0.0, 0.0],
             [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0]],
            columns=["a0", "ac", "as"],
            index=["a0", "ac", "as"],
        )

        self.Q = process_var * self.P.copy()  # bruit de process
        self.R = meas_var                     # bruit de mesure scalaire

        self.last_prob = 0.5

    def _H(self, t):
        """Vecteur d'observation H_t pour le temps t."""
        c = math.cos(self.omega * t)
        s = math.sin(self.omega * t)
        return pd.Series([1.0, c, s], index=["a0", "ac", "as"])

    def predict(self):
        """Kalman prévision: θ ne change pas (process statique), P augmente."""
        self.P = self.P + self.Q

    def update(self, t, z):
        """
        Mise à jour avec observation z (reward moyen ∈ [0,1]) au temps t.
        """
        H = self._H(t)                    # shape (3,)
        h = float(H @ self.theta)         # prédiction observée
        # variance prédite de l'observation
        S = float(H @ self.P @ H) + self.R
        # gain de Kalman
        K = (self.P @ H) / S              # shape (3,)

        # mise à jour état/covariance
        y = z - h                         # innovation
        self.theta = self.theta + K * y
        
        # Correction: K et H doivent être transformés en matrices compatibles
        K_matrix = K.values.reshape(-1, 1)  # (3, 1)
        H_matrix = H.values.reshape(1, -1)  # (1, 3)
        
        # Mise à jour de la matrice de covariance: P = P - K * H * P
        self.P = self.P - pd.DataFrame(
            K_matrix @ H_matrix @ self.P.values,
            index=self.P.index,
            columns=self.P.columns
        )

        # clamp léger pour éviter des probs débiles
        prob = self.prob(t)
        self.last_prob = prob

    def prob(self, t):
        """Probabilité prédite au temps t."""
        H = self._H(t)
        p = float(H @ self.theta)
        # on borne dans [0,1]
        return max(0.01, min(0.99, p))

    def prob_var(self, t):
        """Variance approximative de la probabilité prédite au temps t."""
        H = self._H(t)
        return float(H @ self.P @ H) + self.R

    def estimate(self, t):
        return self.prob(t)

    def slope(self, t_prev, t_curr):
        """Approximation slope: diff de prob entre t_prev et t_curr."""
        p_prev = self.prob(t_prev)
        p_curr = self.prob(t_curr)
        return p_curr - p_prev


# ============================================================
#              STRATÉGIE KALMAN PÉRIODIQUE + UCB
# ============================================================

class PeriodicKalmanStrategy:
    def __init__(self):
        self.client = SphinxAPIClient()
        # périodes connues
        self.T = [10, 20, 200]

        # paramètres de process : plus large pour périodes courtes
        process_vars = [0.01, 0.006, 0.003]
        meas_var = 0.20

        self.planets = [
            PeriodicKalmanPlanet(self.T[i], process_var=process_vars[i], meas_var=meas_var)
            for i in range(3)
        ]

        self.step = 0          # temps = nombre de trips (API calls)
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

            planet.predict()  # augmente juste la variance

            batch = 1
            result = self.client.send_morties(p, batch)
            survived = result["survived"]
            reward = survived / batch

            planet.update(t, reward)
            est = planet.estimate(t)
            sl = 0.0  # slope pas vraiment défini pendant découverte précoce

            self.log_rows.append({
                "step": self.step,
                "sent": self.morties_sent,
                "planet": p,
                "estimate": est,
                "slope": sl,
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
        bonus = 0.8 * math.sqrt(max(var, 1e-6))

        # légère pénalité/bonus possible selon période (optionnel)
        # ici on met un petit nerf pour T=10 très chaotique
        if self.T[p] == 10:
            bonus *= 0.7

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
        # exploiter plus si prob élevée
        if est > 0.70 and sl >= 0:
            return 3
        if est > 0.55:
            return 2
        return 1

    # --------------------------------------------------------
    def run(self):
        print("STRATÉGIE KALMAN PÉRIODIQUE + UCB")

        # EXPLORATION MINIMALE (5 samples au lieu de 20-40!)
        # Juste pour initialiser le filtre, UCB fera le reste
        self.discover_planet(0, samples=5)
        self.discover_planet(1, samples=5)
        self.discover_planet(2, samples=5)
        print(f"Exploration minimale terminée: {self.morties_sent} Morties utilisés")
        print(f"   Il reste {1000 - self.morties_sent} Morties pour l'exploitation!")

        # -------- EXPLOITATION PRINCIPALE --------
        while self.morties_sent < 1000:
            p = self.choose_planet()
            planet = self.planets[p]

            t_prev = self.step - 1
            t_curr = self.step

            planet.predict()
            est_before = planet.estimate(t_curr)
            sl = planet.slope(t_prev, t_curr) if self.step > 0 else 0.0

            batch = self.choose_batch(est_before, sl)
            batch = min(batch, 1000 - self.morties_sent)
            if batch <= 0:
                break

            result = self.client.send_morties(p, batch)
            survived = result["survived"]
            reward = survived / batch

            planet.update(t_curr, reward)
            est_after = planet.estimate(t_curr)
            sl_after = planet.slope(t_prev, t_curr) if self.step > 0 else 0.0

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
        plt.title(f"Run {run_id:03d} (Kalman périodique)")
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
