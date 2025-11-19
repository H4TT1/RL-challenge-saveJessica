import os
import sys
import numpy as np
from collections import deque
from api_client import SphinxAPIClient

class AdaptivePhaseDetector:
    """
    Detects planetary phase patterns through observation and autocorrelation analysis.
    More robust than hardcoded periods - learns from actual survival patterns.
    """
    
    def __init__(self, expected_period: int, max_history: int = 500):
        self.expected_period = expected_period
        self.history = deque(maxlen=max_history)
        self.phase_stats = {}  # phase_idx -> {survived: int, total: int}
        self.trips = 0
        self.confirmed_period = None
        self.confidence = 0.0
        
    def record_outcome(self, survived: bool):
        """Record the outcome of a trip."""
        self.history.append(1 if survived else 0)
        self.trips += 1
        
        # Update phase statistics
        if self.confirmed_period:
            phase = self.trips % self.confirmed_period
        else:
            phase = self.trips % self.expected_period
            
        if phase not in self.phase_stats:
            self.phase_stats[phase] = {"survived": 0, "total": 0}
        
        self.phase_stats[phase]["survived"] += (1 if survived else 0)
        self.phase_stats[phase]["total"] += 1
        
    def detect_period(self, min_samples: int = 30):
        """
        Use autocorrelation to detect the true period.
        Returns (period, confidence) or (None, 0.0) if insufficient data.
        """
        if len(self.history) < min_samples:
            return None, 0.0
        
        data = np.array(list(self.history))
        n = len(data)
        
        # Test periods around expected value
        test_range = range(
            max(5, self.expected_period - 5),
            min(n // 2, self.expected_period + 5)
        )
        
        best_period = self.expected_period
        best_score = -1.0
        
        for period in test_range:
            if n < 2 * period:
                continue
                
            # Calculate autocorrelation at this lag
            mean = np.mean(data)
            c0 = np.sum((data - mean) ** 2)
            
            if c0 == 0:
                continue
                
            # Autocorrelation at lag = period
            ct = np.sum((data[:-period] - mean) * (data[period:] - mean))
            autocorr = ct / c0
            
            if autocorr > best_score:
                best_score = autocorr
                best_period = period
        
        # Confidence based on autocorrelation strength and sample size
        confidence = min(1.0, best_score * (n / (3 * self.expected_period)))
        
        return best_period, confidence
    
    def update_period_estimate(self):
        """Periodically refine period estimate."""
        if self.trips % 20 == 0 and self.trips >= 30:
            period, conf = self.detect_period()
            if conf > 0.5 and period:
                if self.confirmed_period != period:
                    # Period changed - rebuild phase stats
                    self.confirmed_period = period
                    self.confidence = conf
                    self._rebuild_phase_stats()
                else:
                    self.confidence = max(self.confidence, conf)
    
    def _rebuild_phase_stats(self):
        """Rebuild phase statistics with new period."""
        if not self.confirmed_period:
            return
            
        old_stats = self.phase_stats.copy()
        self.phase_stats = {}
        
        # Re-map history to new phases
        for i, outcome in enumerate(list(self.history)[-self.trips:]):
            phase = (i + 1) % self.confirmed_period
            if phase not in self.phase_stats:
                self.phase_stats[phase] = {"survived": 0, "total": 0}
            self.phase_stats[phase]["survived"] += outcome
            self.phase_stats[phase]["total"] += 1
    
    def get_phase_survival_rate(self, lookahead: int = 0):
        """
        Get survival rate for current phase + lookahead.
        Returns (survival_rate, sample_size, confidence).
        """
        period = self.confirmed_period or self.expected_period
        future_phase = (self.trips + lookahead) % period
        
        if future_phase not in self.phase_stats or self.phase_stats[future_phase]["total"] == 0:
            # Unknown phase - use conservative prior
            return 0.5, 0, 0.0
        
        stats = self.phase_stats[future_phase]
        survival_rate = stats["survived"] / stats["total"]
        
        # Confidence increases with sample size
        phase_confidence = min(1.0, stats["total"] / 5.0) * self.confidence
        
        return survival_rate, stats["total"], phase_confidence
    
    def get_current_phase(self):
        """Get current phase index."""
        period = self.confirmed_period or self.expected_period
        return self.trips % period
    
    def get_phase_quality_score(self, lookahead: int = 0):
        """
        Combined score considering survival rate and confidence.
        Higher is better.
        """
        survival_rate, samples, confidence = self.get_phase_survival_rate(lookahead)
        
        # Weight by confidence - unknown phases get neutral score
        if samples == 0:
            return 0.5
        
        # Combine survival rate with confidence
        return survival_rate * (0.7 + 0.3 * confidence)


class ImprovedPhaseAwareStrategy:
    def __init__(self, client: SphinxAPIClient):
        self.client = client
        
        # Initialize detectors with expected periods
        self.detectors = {
            0: AdaptivePhaseDetector(expected_period=10),
            1: AdaptivePhaseDetector(expected_period=20),
            2: AdaptivePhaseDetector(expected_period=200)
        }
        
        self.safe_threshold = 0.70
        self.high_confidence_threshold = 0.80
        
    def run(self):
        print("Improved Phase-Aware Rescue with Adaptive Detection")
        print("=" * 70)
        
        self.client.start_episode()
        status = self.client.get_status()
        morties = status["morties_in_citadel"]
        
        print("\nPhase 1: Exploration (30 trips - learning phase patterns)")
        for i in range(30):
            if morties <= 0:
                break
                
            planet = i % 3
            result = self.client.send_morties(planet, 1)
            survived = result["survived"]
            
            self.detectors[planet].record_outcome(survived)
            self.detectors[planet].update_period_estimate()
            
            morties = result["morties_in_citadel"]
        
        # Report detected periods
        print("\nDetected Periods:")
        for planet in [0, 1, 2]:
            detector = self.detectors[planet]
            period = detector.confirmed_period or detector.expected_period
            conf = detector.confidence
            print(f"   Planet {planet}: Period={period} (confidence: {conf:.2f})")
        
        print("\nPhase 2: Exploitation (using learned patterns)")
        step_count = 30
        
        while morties > 0:
            # Evaluate all planets for next trip
            planet_scores = []
            for planet in [0, 1, 2]:
                detector = self.detectors[planet]
                quality_score = detector.get_phase_quality_score(lookahead=0)
                survival_rate, samples, conf = detector.get_phase_survival_rate(lookahead=0)
                
                planet_scores.append({
                    'planet': planet,
                    'score': quality_score,
                    'survival_rate': survival_rate,
                    'confidence': conf,
                    'samples': samples
                })
            
            # Choose best planet
            best = max(planet_scores, key=lambda x: x['score'])
            planet = best['planet']
            survival_rate = best['survival_rate']
            confidence = best['confidence']
            
            # Adaptive group sizing
            if survival_rate >= self.high_confidence_threshold and confidence > 0.7:
                # Very safe - send larger groups
                group_size = min(5, morties)
            elif survival_rate >= self.safe_threshold and confidence > 0.5:
                # Moderately safe
                group_size = min(3, morties)
            elif morties <= 3:
                # Running low - be conservative
                group_size = 1
            else:
                # Uncertain or risky
                group_size = min(2, morties)
            
            # Send mission
            result = self.client.send_morties(planet, group_size)
            survived = result["survived"]
            
            # Update detector
            self.detectors[planet].record_outcome(survived)
            self.detectors[planet].update_period_estimate()
            
            morties = result["morties_in_citadel"]
            step_count += 1
            
            # Progress updates
            if step_count % 50 == 0:
                saved = result["morties_on_planet_jessica"]
                phase = self.detectors[planet].get_current_phase()
                period = self.detectors[planet].confirmed_period or self.detectors[planet].expected_period
                print(f"[Step {step_count}] Saved: {saved} | P{planet} (phase {phase}/{period}) | "
                      f"Group: {group_size} | Surv: {survival_rate:.2f}")

        final = self.client.get_status()
        saved = final["morties_on_planet_jessica"]
        rate = saved / 1000 * 100
        
        print("\n" + "=" * 70)
        print(f"MISSION COMPLETE")
        print(f"   Morties Saved: {saved} / 1000 ({rate:.1f}%)")
        print(f"\nFinal Phase Statistics:")
        
        for planet in [0, 1, 2]:
            detector = self.detectors[planet]
            period = detector.confirmed_period or detector.expected_period
            print(f"\n   Planet {planet} (Period: {period}):")
            
            # Show phase survival rates
            phase_data = []
            for phase in range(period):
                if phase in detector.phase_stats:
                    stats = detector.phase_stats[phase]
                    if stats["total"] > 0:
                        rate = stats["survived"] / stats["total"]
                        phase_data.append((phase, rate, stats["total"]))
            
            # Sort by phase
            phase_data.sort()
            
            # Show top safe and unsafe phases
            if phase_data:
                safe_phases = [p for p in phase_data if p[1] >= 0.7]
                unsafe_phases = [p for p in phase_data if p[1] < 0.5]
                
                if safe_phases:
                    print(f"      Safe phases: {[f'{p[0]}({p[1]:.0%})' for p in safe_phases[:5]]}")
                if unsafe_phases:
                    print(f"      Risky phases: {[f'{p[0]}({p[1]:.0%})' for p in unsafe_phases[:5]]}")
        
        print("=" * 70)


if __name__ == "__main__":
    try:
        client = SphinxAPIClient()
        strategy = ImprovedPhaseAwareStrategy(client)
        strategy.run()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()