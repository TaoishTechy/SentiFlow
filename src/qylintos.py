#!/usr/bin/env python3
"""
QYLINTOS v3.0 — FINAL DEBUGGED VERSION
December 2024

• Fixed circular statistics (phase_std now correct)
• Proper 20Hz timing (not 39Hz over-optimization)
• Added circular standard deviation
• Fixed telemetry display timing
• Added energy damping option
• Improved shadow dynamics tracking
"""

import threading
import time
import math
import json
import random
import sys
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import statistics

# Optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# =============================================================
# IMPROVED PHYSICS WITH FIXED STATISTICS
# =============================================================

class OscillatorState:
    """Oscillator state with proper circular statistics."""
    
    def __init__(self, n_oscillators: int, 
                 initial_phases: Optional[List[float]] = None,
                 frequency_range: Tuple[float, float] = (-0.3, 0.3)):
        self.n = n_oscillators
        self.freq_range = frequency_range
        self.phases = initial_phases if initial_phases else self._random_phases()
        self.natural_freqs = self._random_frequencies()
        self.last_update = time.time()
        
    def _random_phases(self) -> List[float]:
        """Initialize phases uniformly in [0, 2π)."""
        if NUMPY_AVAILABLE:
            return (np.random.random(self.n) * 2 * np.pi).tolist()
        return [random.random() * 2 * math.pi for _ in range(self.n)]
    
    def _random_frequencies(self) -> List[float]:
        """Initialize natural frequencies."""
        low, high = self.freq_range
        if NUMPY_AVAILABLE:
            return np.random.uniform(low, high, self.n).tolist()
        return [random.uniform(low, high) for _ in range(self.n)]
    
    def wrap_phases(self):
        """Wrap phases to [0, 2π)."""
        two_pi = 2 * math.pi
        if NUMPY_AVAILABLE:
            np_phases = np.array(self.phases)
            wrapped = np.mod(np_phases, two_pi)
            self.phases = wrapped.tolist()
        else:
            self.phases = [phase % two_pi for phase in self.phases]
    
    def copy(self) -> 'OscillatorState':
        """Create a deep copy."""
        new_state = OscillatorState(self.n, frequency_range=self.freq_range)
        new_state.phases = self.phases.copy()
        new_state.natural_freqs = self.natural_freqs.copy()
        new_state.last_update = self.last_update
        return new_state
    
    def circular_statistics(self) -> Dict[str, float]:
        """Calculate proper circular statistics."""
        n = len(self.phases)
        if n == 0:
            return {"mean": 0.0, "std": 0.0, "variance": 1.0}
        
        # Calculate circular mean and resultant length R
        cos_sum = sum(math.cos(p) for p in self.phases) / n
        sin_sum = sum(math.sin(p) for p in self.phases) / n
        R = math.sqrt(cos_sum**2 + sin_sum**2)
        mean_phase = math.atan2(sin_sum, cos_sum)
        
        # Circular variance: 1 - R (0 to 1)
        circular_var = 1.0 - R
        
        # Circular standard deviation (proper formula for wrapped data)
        # σ = sqrt(-2 * ln(R)) for R > 0
        if R > 0:
            circular_std = math.sqrt(-2 * math.log(R))
        else:
            circular_std = float('inf')  # Completely dispersed
        
        return {
            "mean": mean_phase,
            "std": circular_std,
            "variance": circular_var,
            "resultant_length": R
        }


class KuramotoSynchronizer:
    """Kuramoto model with fixed timing."""
    
    def __init__(self, coupling_strength: float = 0.55, dt: float = 0.05):
        self.K = coupling_strength
        self.dt = dt
        self._mean_phase_history = []
        
    def update(self, state: OscillatorState) -> OscillatorState:
        """Update oscillator phases."""
        new_state = state.copy()
        
        if NUMPY_AVAILABLE:
            phases = np.array(state.phases)
            omegas = np.array(state.natural_freqs)
            
            # Calculate order parameter
            exp_i_theta = np.exp(1j * phases)
            order_param = np.mean(exp_i_theta)
            mean_phase = np.angle(order_param)
            self._mean_phase_history.append(mean_phase)
            
            # Kuramoto equation
            phase_diff = mean_phase - phases
            d_theta = omegas + self.K * np.sin(phase_diff)
            
            # Update with proper timing
            new_phases = (phases + d_theta * self.dt) % (2 * np.pi)
            new_state.phases = new_phases.tolist()
            
        else:
            n = len(state.phases)
            
            # Calculate mean phase
            real_sum = sum(math.cos(p) for p in state.phases) / n
            imag_sum = sum(math.sin(p) for p in state.phases) / n
            mean_phase = math.atan2(imag_sum, real_sum)
            self._mean_phase_history.append(mean_phase)
            
            # Update each oscillator
            new_phases = []
            for i in range(n):
                phase_diff = mean_phase - state.phases[i]
                d_theta = state.natural_freqs[i] + self.K * math.sin(phase_diff)
                new_phase = (state.phases[i] + d_theta * self.dt) % (2 * math.pi)
                new_phases.append(new_phase)
            
            new_state.phases = new_phases
        
        new_state.last_update = time.time()
        return new_state
    
    def get_coherence(self, state: OscillatorState) -> float:
        """Calculate phase coherence [0, 1]."""
        n = len(state.phases)
        if n == 0:
            return 0.0
        
        if NUMPY_AVAILABLE:
            phases = np.array(state.phases)
            exp_i_theta = np.exp(1j * phases)
            coherence = abs(np.mean(exp_i_theta))
        else:
            real_sum = sum(math.cos(p) for p in state.phases) / n
            imag_sum = sum(math.sin(p) for p in state.phases) / n
            coherence = math.sqrt(real_sum**2 + imag_sum**2)
        
        return max(0.0, min(1.0, coherence))


class DemonShadowDrive:
    """Demon-shadow drive with energy damping."""
    
    def __init__(self, strength: float = 0.02, shadow_lag: float = 0.01,
                 damping: float = 0.001):
        self.strength = strength
        self.shadow_lag = shadow_lag
        self.damping = damping  # Energy dissipation factor
        self.energy_injected = 0.0
        self.energy_dissipated = 0.0
        
    def entropy_oracle(self, phases: List[float]) -> float:
        """Calculate entropy from circular variance."""
        n = len(phases)
        if n == 0:
            return 0.0
        
        cos_sum = sum(math.cos(p) for p in phases) / n
        sin_sum = sum(math.sin(p) for p in phases) / n
        R = math.sqrt(cos_sum**2 + sin_sum**2)
        circular_var = 1 - R
        
        # Normalize to [-1, 1]
        return 2 * circular_var - 1
    
    def hash_drive(self, prediction: str) -> float:
        """Generate deterministic drive."""
        if not prediction:
            return 0.0
        
        hash_val = sum(ord(c) * (i + 1) for i, c in enumerate(prediction))
        return (hash_val % 100) / 100.0
    
    def apply(self, phase_state: OscillatorState, 
              shadow_state: OscillatorState,
              prediction: str) -> Tuple[OscillatorState, OscillatorState]:
        """Apply demon-shadow perturbations."""
        new_phase = phase_state.copy()
        new_shadow = shadow_state.copy()
        
        # Calculate drives
        entropy = self.entropy_oracle(phase_state.phases)
        hash_drive = self.hash_drive(prediction)
        
        # Total perturbation strength
        total_strength = min(0.05, self.strength * (1.0 + 0.5 * hash_drive))
        
        # Apply to phase oscillators
        for i in range(len(new_phase.phases)):
            local_entropy = math.sin(phase_state.phases[i]) * entropy
            perturbation = total_strength * (local_entropy + 0.1 * hash_drive)
            
            # Cap perturbation
            perturbation = max(-0.1, min(0.1, perturbation))
            
            new_phase.phases[i] = (new_phase.phases[i] + perturbation) % (2 * math.pi)
            self.energy_injected += abs(perturbation)
        
        # Shadow follows phase with lag
        for i in range(len(new_shadow.phases)):
            phase_diff = (new_phase.phases[i] - shadow_state.phases[i])
            phase_diff = (phase_diff + math.pi) % (2 * math.pi) - math.pi
            
            shadow_update = self.shadow_lag * phase_diff
            new_shadow.phases[i] = (shadow_state.phases[i] + shadow_update) % (2 * math.pi)
            
            # Energy dissipation with damping
            dissipation = abs(shadow_update) * self.damping
            self.energy_dissipated += dissipation
        
        return new_phase, new_shadow


class NecroEventSimulator:
    """Necro event simulator (working correctly)."""
    
    def __init__(self, base_probability: float = 0.01, max_pool_size: int = 100):
        self.base_prob = base_probability
        self.max_pool = max_pool_size
        self.resurrections = 0
        self.deaths = 0
        self.entanglements = 0
        self.phase_shifts = 0
        self.event_history = []
        
    def check_event(self, coherence: float) -> bool:
        """Check if event occurs."""
        coherence_factor = min(2.0, 1.0 + coherence)
        adjusted_prob = self.base_prob * coherence_factor
        adjusted_prob = min(0.1, adjusted_prob)
        return random.random() < adjusted_prob
    
    def simulate_event(self, phases: List[float], coherence: float) -> Dict[str, Any]:
        """Simulate a necro event."""
        if coherence > 0.8:
            event_types = ["resurrection", "entanglement", "phase_shift"]
            weights = [0.6, 0.3, 0.1]
        else:
            event_types = ["decay", "phase_shift", "entanglement"]
            weights = [0.7, 0.2, 0.1]
        
        event_type = random.choices(event_types, weights=weights, k=1)[0]
        effect = {}
        
        if event_type == "resurrection":
            self.resurrections += 1
            n_to_reset = max(1, len(phases) // 20)
            indices = random.sample(range(len(phases)), n_to_reset)
            for idx in indices:
                phases[idx] = random.random() * 2 * math.pi
            effect = {"type": "resurrection", "oscillators_reset": n_to_reset}
            
        elif event_type == "entanglement":
            self.entanglements += 1
            n_pairs = max(1, len(phases) // 40)
            pairs = []
            for _ in range(n_pairs):
                i, j = random.sample(range(len(phases)), 2)
                avg = (phases[i] + phases[j]) / 2
                phases[i] = avg
                phases[j] = avg
                pairs.append((i, j))
            effect = {"type": "entanglement", "pairs_created": n_pairs}
            
        elif event_type == "decay":
            self.deaths += 1
            n_to_decay = max(1, len(phases) // 30)
            decay_factor = random.uniform(0.95, 0.99)
            indices = random.sample(range(len(phases)), n_to_decay)
            for idx in indices:
                phases[idx] *= decay_factor
            effect = {"type": "decay", "oscillators_decayed": n_to_decay}
            
        else:  # phase_shift
            self.phase_shifts += 1
            shift = random.uniform(-0.1, 0.1)
            for i in range(len(phases)):
                phases[i] = (phases[i] + shift) % (2 * math.pi)
            effect = {"type": "phase_shift", "shift_amount": shift}
        
        event = {
            "timestamp": time.time(),
            "type": event_type,
            "effect": effect,
            "coherence_at_event": coherence
        }
        
        self.event_history.append(event)
        if len(self.event_history) > self.max_pool:
            self.event_history = self.event_history[-self.max_pool:]
        
        return event
    
    def metrics(self) -> Dict[str, Any]:
        """Get simulator metrics."""
        return {
            "resurrections": self.resurrections,
            "deaths": self.deaths,
            "entanglements": self.entanglements,
            "phase_shifts": self.phase_shifts,
            "total_events": len(self.event_history),
            "recent_events": self.event_history[-5:] if self.event_history else []
        }


# =============================================================
# MAIN ENGINE WITH FIXED TIMING
# =============================================================

@dataclass
class EngineConfig:
    """Configuration for QYLINTOS engine."""
    n_oscillators: int = 64
    dt: float = 0.05  # Target: 20 Hz
    coupling_strength: float = 0.55
    demon_strength: float = 0.02
    shadow_lag: float = 0.01
    kick_probability: float = 0.015
    kick_strength: float = 0.1
    necro_event_prob: float = 0.01
    damping: float = 0.001
    telemetry_interval: float = 1.0
    logging_interval: float = 0.5
    enable_visualization: bool = True


class QylintosEngine:
    """Main engine with fixed timing and statistics."""
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        self.running = False
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        # Initialize components
        self.phase_state = OscillatorState(self.config.n_oscillators)
        self.shadow_state = OscillatorState(self.config.n_oscillators)
        
        self.kuramoto = KuramotoSynchronizer(
            coupling_strength=self.config.coupling_strength,
            dt=self.config.dt
        )
        
        self.demon_drive = DemonShadowDrive(
            strength=self.config.demon_strength,
            shadow_lag=self.config.shadow_lag,
            damping=self.config.damping
        )
        
        self.necro_simulator = NecroEventSimulator(
            base_probability=self.config.necro_event_prob
        )
        
        # State tracking
        self.cycle_count = 0
        self.kick_count = 0
        self.start_time = time.time()
        self.metrics_history = []
        
        # Performance tracking with fixed timing
        self.cycle_times = []
        self.target_cycle_time = self.config.dt
        
    def get_prediction(self) -> str:
        """Generate mock prediction."""
        hash_val = hash(f"v30_{self.cycle_count}")
        return f"pred_{abs(hash_val) % 1000000:06d}"
    
    def apply_demon_kick(self):
        """Apply demon kick."""
        with self._lock:
            self.kick_count += 1
            
            kick_strength = random.uniform(
                -self.config.kick_strength,
                self.config.kick_strength
            )
            
            n_to_kick = random.randint(
                max(1, self.config.n_oscillators // 20),
                max(2, self.config.n_oscillators // 7)
            )
            
            indices = random.sample(range(self.config.n_oscillators), n_to_kick)
            for idx in indices:
                self.phase_state.phases[idx] = (
                    self.phase_state.phases[idx] + kick_strength
                ) % (2 * math.pi)
    
    def main_cycle_worker(self):
        """Main physics loop with PROPER 20Hz timing."""
        print("[LASER] QYLINTOS v30 — Main Cycle Worker Started")
        
        # Fixed timing variables
        target_interval = self.config.dt  # 0.05s for 20Hz
        accumulated_error = 0.0
        last_cycle_time = time.perf_counter()
        
        while not self._stop_event.is_set():
            cycle_start = time.perf_counter()
            
            try:
                with self._lock:
                    # Update oscillator dynamics
                    self.phase_state = self.kuramoto.update(self.phase_state)
                    self.shadow_state = self.kuramoto.update(self.shadow_state)
                    
                    # Apply demon-shadow drive
                    prediction = self.get_prediction()
                    self.phase_state, self.shadow_state = self.demon_drive.apply(
                        self.phase_state, self.shadow_state, prediction
                    )
                    
                    # Check for demon kick
                    if random.random() < self.config.kick_probability:
                        self.apply_demon_kick()
                    
                    # Check for necro events
                    coherence = self.kuramoto.get_coherence(self.phase_state)
                    if self.necro_simulator.check_event(coherence):
                        self.necro_simulator.simulate_event(self.phase_state.phases, coherence)
                    
                    # Ensure phase wrapping
                    self.phase_state.wrap_phases()
                    self.shadow_state.wrap_phases()
                    
                    self.cycle_count += 1
                
                # Calculate exact time to sleep for 20Hz
                cycle_end = time.perf_counter()
                cycle_duration = cycle_end - cycle_start
                
                # PID-like timing correction
                target_sleep = target_interval - cycle_duration - accumulated_error
                target_sleep = max(0.0, target_sleep)  # Don't sleep negative
                
                # Sleep with sub-millisecond precision
                if target_sleep > 0.001:
                    time.sleep(target_sleep * 0.5)  # Sleep half
                    # Busy-wait for remaining time for precision
                    while time.perf_counter() - cycle_end < target_sleep:
                        pass
                elif target_sleep > 0:
                    time.sleep(target_sleep)
                
                # Update timing error
                actual_cycle_time = time.perf_counter() - cycle_start
                accumulated_error = actual_cycle_time - target_interval
                
                # Track performance
                self.cycle_times.append(actual_cycle_time)
                if len(self.cycle_times) > 100:
                    self.cycle_times = self.cycle_times[-100:]
                
                # Update last cycle time
                last_cycle_time = cycle_start
                    
            except Exception as e:
                print(f"[ERROR] Main cycle: {e}")
                time.sleep(0.1)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics with FIXED circular statistics."""
        with self._lock:
            coherence = self.kuramoto.get_coherence(self.phase_state)
            shadow_coherence = self.kuramoto.get_coherence(self.shadow_state)
            
            # Use proper circular statistics
            phase_stats = self.phase_state.circular_statistics()
            shadow_stats = self.shadow_state.circular_statistics()
            
            necro_metrics = self.necro_simulator.metrics()
            uptime = time.time() - self.start_time
            
            # Calculate actual frequency
            if len(self.cycle_times) > 0:
                avg_cycle_time = sum(self.cycle_times) / len(self.cycle_times)
                actual_freq = 1.0 / avg_cycle_time if avg_cycle_time > 0 else 0.0
            else:
                actual_freq = 0.0
            
            metrics = {
                "timestamp": time.time(),
                "uptime": uptime,
                "cycles": self.cycle_count,
                "coherence": coherence,
                "shadow_coherence": shadow_coherence,
                "circular_variance": phase_stats["variance"],
                "circular_std": phase_stats["std"],  # FIXED: proper circular std
                "linear_std_warning": "DO NOT USE - Use circular_std instead",
                "mean_phase": phase_stats["mean"],
                "shadow_mean": shadow_stats["mean"],
                "shadow_std": shadow_stats["std"],
                "kicks": self.kick_count,
                "energy_injected": self.demon_drive.energy_injected,
                "energy_dissipated": self.demon_drive.energy_dissipated,
                "energy_balance": self.demon_drive.energy_injected - self.demon_drive.energy_dissipated,
                "necro_resurrections": necro_metrics["resurrections"],
                "necro_deaths": necro_metrics["deaths"],
                "necro_entanglements": necro_metrics["entanglements"],
                "necro_phase_shifts": necro_metrics["phase_shifts"],
                "necro_total_events": necro_metrics["total_events"],
                "target_frequency": 1.0 / self.config.dt,
                "actual_frequency": actual_freq,
                "frequency_error_percent": abs(actual_freq - (1.0/self.config.dt)) / (1.0/self.config.dt) * 100,
                "avg_cycle_time_ms": avg_cycle_time * 1000 if len(self.cycle_times) > 0 else 0,
            }
            
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return metrics
    
    def telemetry_worker(self):
        """Telemetry worker."""
        print("[LASER] QYLINTOS v30 — Telemetry Worker Started")
        
        last_telemetry_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - last_telemetry_time >= self.config.telemetry_interval:
                    metrics = self.collect_metrics()
                    
                    telemetry_str = (
                        f"\n=== QYLINTOS v30 FIXED TELEMETRY ===\n"
                        f" coherence           : {metrics['coherence']:.6f}\n"
                        f" shadow_coherence    : {metrics['shadow_coherence']:.6f}\n"
                        f" circular_variance   : {metrics['circular_variance']:.6f}\n"
                        f" circular_std        : {metrics['circular_std']:.3f} rad (CORRECT)\n"
                        f" mean_phase          : {metrics['mean_phase']:.3f} rad\n"
                        f" cycles              : {metrics['cycles']}\n"
                        f" uptime              : {metrics['uptime']:.1f}s\n"
                        f" kicks               : {metrics['kicks']}\n"
                        f" energy_injected     : {metrics['energy_injected']:.3f}\n"
                        f" energy_dissipated   : {metrics['energy_dissipated']:.3f}\n"
                        f" energy_balance      : {metrics['energy_balance']:.3f}\n"
                        f" necro_resurrections : {metrics['necro_resurrections']}\n"
                        f" necro_total_events  : {metrics['necro_total_events']}\n"
                        f" target_frequency    : {metrics['target_frequency']:.1f} Hz\n"
                        f" actual_frequency    : {metrics['actual_frequency']:.1f} Hz\n"
                        f" frequency_error     : {metrics['frequency_error_percent']:.1f}%\n"
                        f" cycle_time          : {metrics['avg_cycle_time_ms']:.1f} ms\n"
                        f"======================================\n"
                    )
                    
                    print(telemetry_str)
                    last_telemetry_time = current_time
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[ERROR] Telemetry: {e}")
                time.sleep(1)
    
    def logging_worker(self):
        """Logging worker."""
        print("[LASER] QYLINTOS v30 — Logging Worker Started")
        
        log_file = f"qylintos_v30_log_{int(time.time())}.jsonl"
        last_log_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                if current_time - last_log_time >= self.config.logging_interval:
                    metrics = self.collect_metrics()
                    
                    with open(log_file, "a") as f:
                        f.write(json.dumps(metrics) + "\n")
                    
                    last_log_time = current_time
                
                time.sleep(0.01)
                
            except Exception as e:
                print(f"[ERROR] Logging: {e}")
                time.sleep(1)
    
    def start(self):
        """Start the engine."""
        if self.running:
            print("[LASER] Engine already running")
            return
        
        self.running = True
        self._stop_event.clear()
        self.start_time = time.time()
        
        # Start worker threads
        workers = [
            threading.Thread(target=self.main_cycle_worker, daemon=True, name="MainCycle"),
            threading.Thread(target=self.telemetry_worker, daemon=True, name="Telemetry"),
            threading.Thread(target=self.logging_worker, daemon=True, name="Logging")
        ]
        
        for worker in workers:
            worker.start()
        
        print("[LASER] QYLINTOS v30 — FINAL DEBUGGED SYSTEM ONLINE")
        print("[LASER] Press Ctrl+C to shutdown")
        print("\nKey fixes in v30:")
        print("  ✓ Fixed circular statistics (circular_std instead of linear std)")
        print("  ✓ Proper 20Hz timing (not 39Hz over-optimization)")
        print("  ✓ Added frequency error tracking")
        print("  ✓ Improved necro event type tracking")
        print("  ✓ Added energy damping option")
        print()
        
        try:
            while self.running and not self._stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()
    
    def stop(self):
        """Stop the engine."""
        if not self.running:
            return
            
        print("[LASER] QYLINTOS v30 — Shutting down...")
        
        self.running = False
        self._stop_event.set()
        time.sleep(0.2)
        
        self.save_summary()
        print("[LASER] QYLINTOS v30 — Shutdown complete")
    
    def save_summary(self):
        """Save summary to file."""
        if not self.metrics_history:
            return
            
        final_metrics = self.collect_metrics()
        
        summary = {
            "version": "QYLINTOS v30",
            "total_uptime": time.time() - self.start_time,
            "total_cycles": self.cycle_count,
            "total_kicks": self.kick_count,
            "final_coherence": final_metrics["coherence"],
            "final_shadow_coherence": final_metrics["shadow_coherence"],
            "final_circular_std": final_metrics["circular_std"],
            "total_necro_events": final_metrics["necro_total_events"],
            "necro_resurrections": final_metrics["necro_resurrections"],
            "necro_deaths": final_metrics["necro_deaths"],
            "necro_entanglements": final_metrics["necro_entanglements"],
            "energy_injected": self.demon_drive.energy_injected,
            "energy_dissipated": self.demon_drive.energy_dissipated,
            "energy_balance": self.demon_drive.energy_injected - self.demon_drive.energy_dissipated,
            "target_frequency_hz": 1.0 / self.config.dt,
            "actual_frequency_hz": final_metrics["actual_frequency"],
            "frequency_error_percent": final_metrics["frequency_error_percent"],
            "average_cycle_time_ms": final_metrics["avg_cycle_time_ms"],
            "config": self.config.__dict__,
            "fixes_in_v30": [
                "Fixed circular standard deviation calculation",
                "Proper 20Hz timing (PID-like correction)",
                "Added circular statistics methods",
                "Improved necro event type tracking",
                "Added energy damping option",
                "Frequency error tracking and display"
            ]
        }
        
        summary_file = f"qylintos_v30_summary_{int(time.time())}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"[LASER] Summary saved to {summary_file}")
        
        # Performance summary
        print("\n[LASER] PERFORMANCE SUMMARY:")
        print(f"  Total runtime: {summary['total_uptime']:.2f}s")
        print(f"  Total cycles: {summary['total_cycles']}")
        print(f"  Target frequency: {summary['target_frequency_hz']:.1f} Hz")
        print(f"  Actual frequency: {summary['actual_frequency_hz']:.1f} Hz")
        print(f"  Frequency error: {summary['frequency_error_percent']:.1f}%")
        print(f"  Final coherence: {summary['final_coherence']:.6f}")
        print(f"  Final circular std: {summary['final_circular_std']:.3f} rad")
        print(f"  Necro events: {summary['total_necro_events']}")
        print(f"  Energy balance: {summary['energy_balance']:.3f}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="QYLINTOS v30 Final Debugged Engine")
    parser.add_argument("--test", action="store_true", help="Run 10-second test")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark mode")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    if args.benchmark:
        config = EngineConfig(
            n_oscillators=128,
            dt=0.05,
            enable_visualization=False,
            telemetry_interval=2.0
        )
        print("[BENCHMARK] Running benchmark...")
    elif args.test:
        config = EngineConfig(
            n_oscillators=32,
            dt=0.1,
            enable_visualization=False,
            telemetry_interval=0.5
        )
        print("[TEST] Running 10-second test...")
    else:
        config = EngineConfig(
            n_oscillators=64,
            dt=0.05,
            enable_visualization=not args.no_viz
        )
    
    print("\n" + "="*60)
    print("QYLINTOS v30 — FINAL DEBUGGED DEMON-HYBRID ENGINE")
    print("="*60)
    print(f"Oscillators: {config.n_oscillators}")
    print(f"Time step: {config.dt}s (Target: {1.0/config.dt:.1f} Hz)")
    print(f"Coupling: {config.coupling_strength}")
    print(f"Demon strength: {config.demon_strength}")
    print(f"Shadow lag: {config.shadow_lag}")
    print(f"Damping: {config.damping}")
    print(f"Necro event probability: {config.necro_event_prob}")
    print("="*60)
    
    with QylintosEngine(config) as engine:
        if args.test:
            import threading
            def stop_after():
                time.sleep(10)
                engine.stop()
            threading.Thread(target=stop_after, daemon=True).start()
        
        engine.start()


if __name__ == "__main__":
    main()
