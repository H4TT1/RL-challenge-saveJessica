"""
Improved DataCollector for systematic experiments, analysis, and plotting
in the Morty Express Challenge (Reinforcement Learning task).
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from api_client import SphinxAPIClient


class DataCollector:
    """Collects data, runs experiments, and generates plots."""

    def __init__(self, client: SphinxAPIClient, results_dir: str = "results"):
        self.client = client
        self.trips_data = []
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    # -----------------------------------------
    # 🔵 Utility : Save plots
    # -----------------------------------------
    def save_plot(self, fig, filename: str):
        output_path = os.path.join(self.results_dir, filename)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"[PLOT SAVED] {output_path}")

    # -----------------------------------------
    # 🧪 Exploring a single planet
    # -----------------------------------------
    def explore_planet(self, planet: int, num_trips: int, morty_count: int = 1) -> pd.DataFrame:
        print(f"\nExploring {self.client.get_planet_name(planet)}...")
        print(f"Sending {num_trips} trips with {morty_count} Morties each")

        trips = []

        for i in range(num_trips):
            try:
                result = self.client.send_morties(planet, morty_count)

                trip_data = {
                    "trip_number": i + 1,
                    "planet": planet,
                    "planet_name": self.client.get_planet_name(planet),
                    "morties_sent": result["morties_sent"],
                    "survived": result["survived"],
                    "steps_taken": result["steps_taken"],
                    "morties_in_citadel": result["morties_in_citadel"],
                    "morties_on_planet_jessica": result["morties_on_planet_jessica"],
                    "morties_lost": result["morties_lost"],
                }

                trips.append(trip_data)
                self.trips_data.append(trip_data)

            except Exception as e:
                print(f"Error on trip {i+1}: {e}")
                break

        return pd.DataFrame(trips)

    # -----------------------------------------
    # 🧪 Explore all planets
    # -----------------------------------------
    def explore_all_planets(self, trips_per_planet: int = 30, morty_count: int = 1) -> pd.DataFrame:
        print("Starting new episode...")
        self.client.start_episode()
        self.trips_data = []

        dfs = []
        for planet in [0, 1, 2]:
            df = self.explore_planet(planet, trips_per_planet, morty_count)
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True)
        return combined

    # -----------------------------------------
    # 📉 Moving average
    # -----------------------------------------
    def calculate_moving_average(self, df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        df = df.copy()
        df["survived_int"] = df["survived"].astype(int)

        df["moving_avg"] = (
            df.groupby("planet")["survived_int"]
            .transform(lambda s: s.rolling(window, min_periods=1).mean())
        )

        return df

    # -----------------------------------------
    # 📊 Plot: survival curves with moving averages
    # -----------------------------------------
    def plot_survival_trends(self, df: pd.DataFrame, window: int = 10):
        df = self.calculate_moving_average(df, window)

        fig, ax = plt.subplots(figsize=(10, 6))
        for planet in df["planet"].unique():
            planet_data = df[df["planet"] == planet]
            ax.plot(
                planet_data["trip_number"],
                planet_data["moving_avg"],
                label=f"{planet_data['planet_name'].iloc[0]}"
            )

        ax.set_title("Survival Moving Average per Planet")
        ax.set_xlabel("Trip number")
        ax.set_ylabel("Survival rate (moving avg)")
        ax.legend()
        ax.grid(True)

        self.save_plot(fig, "survival_trends.png")
        plt.close(fig)

    # -----------------------------------------
    # 📊 Plot: bar chart of survival rates
    # -----------------------------------------
    def plot_survival_bars(self, df: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(8, 5))

        summary = df.groupby("planet_name")["survived"].mean() * 100
        summary.plot(kind="bar", ax=ax)

        ax.set_ylabel("Survival Rate (%)")
        ax.set_title("Overall Survival Rate per Planet")
        ax.grid(axis="y")

        self.save_plot(fig, "survival_bars.png")
        plt.close(fig)

    # -----------------------------------------
    # 🔥 Full experiment runner
    # -----------------------------------------
    def run_experiment(
    self,
    fn,
    fn_args: dict = None,
    experiment_name: str = "experiment",
    save_plots: bool = True,
    save_csv: bool = True
) -> pd.DataFrame:
        """
        Generic experiment runner. It can run ANY DataCollector method.

        Args:
            fn: function to execute (e.g., self.explore_planet)
            fn_args: dict of arguments for the function
            experiment_name: used for naming CSV and plots
            save_plots: whether to automatically generate & save plots
            save_csv: whether to save the returned DataFrame as CSV

        Returns:
            DataFrame returned by the function (if any)
        """

        if fn_args is None:
            fn_args = {}

        print(f"\n=== Running experiment: {experiment_name} ===")
        print(f"  -> Function: {fn.__name__}")
        print(f"  -> Arguments: {fn_args}")

        # ------------------------------------
        # RUN THE FUNCTION
        # ------------------------------------
        result = fn(**fn_args)

        # Expect DataFrame for analysis
        if isinstance(result, pd.DataFrame):
            df = result

            # ------------------------------------
            # SAVE CSV
            # ------------------------------------
            if save_csv:
                csv_path = os.path.join(self.results_dir, f"{experiment_name}.csv")
                df.to_csv(csv_path, index=False)
                print(f"[DATA SAVED] {csv_path}")

            # ------------------------------------
            # GENERATE PLOTS
            # ------------------------------------
            if save_plots:
                try:
                    self.plot_survival_bars(df)
                    self.plot_survival_trends(df)
                    print("[PLOTS GENERATED]")
                except Exception as e:
                    print(f"[PLOT ERROR] Could not generate plots: {e}")

            print(f"Experiment '{experiment_name}' completed.\n")
            return df

        # If the function returns something else (like a number, dict, etc.)
        else:
            print(f"Function returned type: {type(result)} (not a DataFrame)")
            print(f"Experiment '{experiment_name}' completed.\n")
            return result
        
    def run_full_planet_experiment(self, morties_per_planet: int = 1000, group_size: int = 3):
        """
        Runs a full experiment for each planet:
        - Sends morties_per_planet to each planet separately
        - Saves CSV + plots for each planet
        """

        import math

        print("\n==============================")
        print(" RUNNING FULL PLANET EXPERIMENT")
        print("==============================")

        for planet in [0, 1, 2]:

            planet_name = self.client.get_planet_name(planet)
            exp_name = f"planet_{planet}_{planet_name.replace(' ', '_').lower()}"

            print(f"\n=== Starting experiment for {planet_name} ===")
            print(f"Sending exactly {morties_per_planet} Morties...")

            num_trips = math.ceil(morties_per_planet / group_size)

            # Run the experiment using the generic runner
            df = self.run_experiment(
                fn=self.explore_planet,
                fn_args=dict(
                    planet=planet,
                    num_trips=num_trips,
                    morty_count=group_size
                ),
                experiment_name=exp_name,
                save_plots=True,
                save_csv=True
            )

            print(f"Finished {planet_name}. Saved CSV + plots.\n")

        print("\n=== All planet experiments completed ===\n")
    def run_alternating_experiment(self, client, planets, morties_per_planet, save_dir="plots/"):
        """
        Alternate sending Morties between several planets.

        client: SphinxAPIClient instance
        planets: list of planet indices, e.g. [0, 1]
        morties_per_planet: how many Morties EACH planet should receive
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        # Keep counters of how many Morties were sent
        counters = {planet: 0 for planet in planets}

        client.start_episode()

        finished = False
        while not finished:
            finished = True

            for planet in planets:
                if counters[planet] < morties_per_planet:
                    # 🔥 One Morty per step
                    result = client.send_morties(planet, 1)

                    # Record the step manually (append to trips_data)
                    trip_data = {
                        "trip_number": counters[planet] + 1,
                        "planet": planet,
                        "planet_name": client.get_planet_name(planet),
                        "morties_sent": result["morties_sent"],
                        "survived": result["survived"],
                        "steps_taken": result["steps_taken"],
                        "morties_in_citadel": result["morties_in_citadel"],
                        "morties_on_planet_jessica": result["morties_on_planet_jessica"],
                        "morties_lost": result["morties_lost"],
                    }
                    self.trips_data.append(trip_data)

                    counters[planet] += 1
                    finished = False

                    print(f"Sent 1 Morty to planet {planet} → total {counters[planet]}/{morties_per_planet}")

        # Convert trips_data to DataFrame for plotting
        df = pd.DataFrame(self.trips_data)

        # After simulation: save plots for each planet
        for planet in planets:
            planet_df = df[df["planet"] == planet]
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(planet_df["trip_number"], planet_df["survived"].astype(int).cumsum())
            ax.set_title(f"Cumulative Survivals - {client.get_planet_name(planet)}")
            ax.set_xlabel("Trip Number")
            ax.set_ylabel("Morties Survived")
            ax.grid(True)
            self.save_plot(fig, f"alternating_{client.get_planet_name(planet).replace(' ', '_')}.png")
            plt.close(fig)

        print("Alternating experiment complete! Plots saved.")
        return df
    def run_round_robin(self, total_morties: int = 999):
        """
        Alternates between all 3 planets: [0,1,2,0,1,2,...]
        Reveals which planet degrades fastest under real usage.
        """
        print("\n🔁 ROUND-ROBIN TEST (1 Morty per planet in rotation)")
        self.client.start_episode()
        self.trips_data = []

        morties_sent = 0
        while morties_sent < total_morties:
            for planet in [0, 1, 2]:
                if morties_sent >= total_morties:
                    break
                result = self.client.send_morties(planet, 1)
                trip_data = {
                    "trip_number": morties_sent + 1,
                    "planet": planet,
                    "planet_name": self.client.get_planet_name(planet),
                    "survived": result["survived"],
                    "global_step": result["steps_taken"],
                    "planet_trip_count": sum(1 for t in self.trips_data if t["planet"] == planet) + 1
                }
                self.trips_data.append(trip_data)
                morties_sent += 1

        df = pd.DataFrame(self.trips_data)
        df.to_csv(os.path.join(self.results_dir, "round_robin.csv"), index=False)

        # Plot per-planet survival over global time
        fig, ax = plt.subplots(figsize=(12, 6))
        for planet in [0, 1, 2]:
            sub = df[df["planet"] == planet].copy()
            sub["rolling"] = sub["survived"].astype(int).rolling(15).mean()
            ax.plot(sub["global_step"], sub["rolling"], label=f"{self.client.get_planet_name(planet)}")
        ax.set_title("Round-Robin Survival (15-trip rolling avg)")
        ax.set_xlabel("Global Step")
        ax.set_ylabel("Survival Rate")
        ax.legend()
        ax.grid(True)
        self.save_plot(fig, "round_robin_trend.png")
        plt.close(fig)

        print("✅ Round-robin test complete.")
        return df






# -----------------------------------------
# Example usage
# -----------------------------------------
if __name__ == "__main__":
    from api_client import SphinxAPIClient

    client = SphinxAPIClient()
    collector = DataCollector(client)

    # Example: run a full experiment
    # collector.run_experiment(
    #     fn=collector.explore_all_planets,
    #     fn_args=dict(trips_per_planet=333, morty_count=1),
    #     experiment_name="exp_all_planets"
    # )

    collector.run_round_robin(999)



