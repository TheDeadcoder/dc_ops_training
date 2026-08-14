"""Physics-outcome scorecard — an evaluation metric deliberately INDEPENDENT
of the training reward.

Why this file exists (review issue 1.4)
---------------------------------------
`evaluate.py` used to report ``sum(obs.reward)`` — the environment reward.
That is the *same* quantity GRPO's ``env_reward_fn`` optimises (``3.0 * r_now``
+ proxy deltas + bonuses). Reporting it back as the headline result is circular:
"we optimised X and X went up" is the definition of training, not evidence of
learning.

This module reports outcomes a datacenter engineer would recognise — peak inlet
temperature, degree-minutes outside the ASHRAE envelope, UPS state-of-charge,
generator-start latency, PUE/energy, and command hygiene. The model was **never
rewarded for any of them**. If a trained model improves on these while an
SFT-only or base model does not, that is real evidence rather than a restatement
of the objective.

It is kept in its own file, with no import of ``rewards.py``, on purpose: the
physical separation from the reward code is itself part of the argument to a
reviewer that the metric is not the objective in disguise.

Data source
-----------
Everything here is computed from the **public** ``DcOpsObservation.metadata``
dict emitted by the environment each step (``sim_time_s``, per-zone
``max_inlet_temp_c``, ``pue``, per-UPS ``battery_soc``/``mode``, generator/
utility state) plus the command string and the environment's ``action_result``
feedback. The one place that needs private simulator state — mapping each zone
to its ASHRAE thresholds — is isolated in :func:`zone_thresholds_from_env`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

# Fallback ASHRAE envelope (class A2) used only when a zone's real thresholds
# could not be resolved from the environment.
_DEFAULT_RECOMMENDED_MAX_C = 27.0
_DEFAULT_ALLOWABLE_MAX_C = 32.0

# Substrings that identify the environment's command-feedback messages.
# Sourced from dc_ops_env/actions/parser.py.
_MALFORMED_TARGET_MARKERS = ("not found",)                 # "Rack 'A-4' not found."
_INVALID_COMMAND_MARKERS = ("unknown command",)            # "Unknown command: 'foo'."
_INVALID_ARG_MARKERS = ("invalid ", "unknown ups mode")    # bad value / bad mode


def zone_thresholds_from_env(env) -> dict[str, tuple[float, float]]:
    """Build ``{zone_id: (recommended_max_c, allowable_max_c)}`` from a reset env.

    Reads ``env._thermal_sim`` (private) because the per-zone ASHRAE class is not
    part of the public observation metadata. Isolated here so the rest of the
    harness stays on the public API; returns ``{}`` if the layout is unavailable,
    in which case the scorecard falls back to the class-A2 envelope.
    """
    try:
        from dc_ops_env.config import ASHRAE_CLASSES

        thermal = getattr(env, "_thermal_sim", None)
        if thermal is None:
            return {}
        out: dict[str, tuple[float, float]] = {}
        for zone in thermal.state.zones:
            ashrae = ASHRAE_CLASSES.get(zone.ashrae_class)
            if ashrae is not None:
                out[zone.zone_id] = (ashrae.recommended_max_c, ashrae.allowable_max_c)
        return out
    except Exception:
        return {}


@dataclass
class PhysicsScorecard:
    """Accumulates physics + operational outcomes over a single episode.

    Call :meth:`observe_reset` once with the initial observation, then
    :meth:`observe_step` after every ``env.step`` with the issued command and the
    resulting observation. Read the results with :meth:`summary`.
    """

    zone_thresholds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # --- accumulators (populated as the episode runs) ----------------------
    _prev_sim_time_s: Optional[float] = None
    peak_inlet_c: float = 0.0
    degree_min_over_recommended: float = 0.0
    degree_min_over_allowable: float = 0.0
    min_ups_soc: Optional[float] = None
    time_on_battery_s: float = 0.0
    _utility_lost_at_s: Optional[float] = None
    _generator_online_at_s: Optional[float] = None
    _pue_time_weighted: float = 0.0
    _pue_time_total_s: float = 0.0
    total_energy_kwh: float = 0.0
    n_commands: int = 0
    n_invalid_command: int = 0
    n_malformed_target: int = 0
    n_invalid_arg: int = 0

    # ----------------------------------------------------------------------
    def observe_reset(self, obs) -> None:
        """Seed the integrators from the initial observation (no dt yet)."""
        meta = getattr(obs, "metadata", None) or {}
        self._prev_sim_time_s = meta.get("sim_time_s")
        self._sample_instant(meta)  # capture t=0 peak/SoC without integrating

    def observe_step(self, command: str, obs) -> None:
        """Fold one post-step observation into the accumulators.

        ``command`` is the raw command string the agent issued; ``obs`` is the
        observation the environment returned for it (so ``obs.action_result``
        is the feedback for exactly this command).
        """
        meta = getattr(obs, "metadata", None) or {}
        now = meta.get("sim_time_s")
        dt_s = 0.0
        if now is not None and self._prev_sim_time_s is not None:
            dt_s = max(0.0, now - self._prev_sim_time_s)
        if now is not None:
            self._prev_sim_time_s = now

        self._classify_command(command, getattr(obs, "action_result", "") or "")
        self._sample_instant(meta)
        self._integrate_interval(meta, dt_s)

    # ----------------------------------------------------------------------
    def _classify_command(self, command: str, action_result: str) -> None:
        from .constants import KNOWN_COMMANDS

        self.n_commands += 1
        head = command.strip().split()[0].lower() if command.strip() else ""
        msg = action_result.lower()

        if head not in KNOWN_COMMANDS or any(m in msg for m in _INVALID_COMMAND_MARKERS):
            self.n_invalid_command += 1
        elif any(m in msg for m in _MALFORMED_TARGET_MARKERS):
            self.n_malformed_target += 1
        elif any(m in msg for m in _INVALID_ARG_MARKERS):
            self.n_invalid_arg += 1

    def _sample_instant(self, meta: dict) -> None:
        """Update running extrema that do not depend on the time interval."""
        for zone_id, zone in (meta.get("zones") or {}).items():
            t = zone.get("max_inlet_temp_c")
            if t is not None and t > self.peak_inlet_c:
                self.peak_inlet_c = t

        power = meta.get("power") or {}
        for unit_id, unit in power.items():
            if not isinstance(unit, dict) or "battery_soc" not in unit:
                continue
            soc = unit["battery_soc"]
            if self.min_ups_soc is None or soc < self.min_ups_soc:
                self.min_ups_soc = soc

    def _integrate_interval(self, meta: dict, dt_s: float) -> None:
        """Accumulate all quantities that are integrated over the interval dt."""
        if dt_s <= 0.0:
            return
        dt_min = dt_s / 60.0
        dt_h = dt_s / 3600.0

        # Degree-minutes outside the ASHRAE envelope (per zone, summed).
        for zone_id, zone in (meta.get("zones") or {}).items():
            t = zone.get("max_inlet_temp_c")
            if t is None:
                continue
            rec, allow = self.zone_thresholds.get(
                zone_id, (_DEFAULT_RECOMMENDED_MAX_C, _DEFAULT_ALLOWABLE_MAX_C)
            )
            self.degree_min_over_recommended += max(0.0, t - rec) * dt_min
            self.degree_min_over_allowable += max(0.0, t - allow) * dt_min

        # Energy + time-weighted PUE.
        it_kw = meta.get("total_it_load_kw") or 0.0
        cool_kw = meta.get("total_cooling_power_kw") or 0.0
        self.total_energy_kwh += (it_kw + cool_kw) * dt_h
        pue = meta.get("pue")
        if pue is not None:
            self._pue_time_weighted += pue * dt_s
            self._pue_time_total_s += dt_s

        # Power: time on battery, and generator-online latency.
        power = meta.get("power") or {}
        now = meta.get("sim_time_s")
        on_battery = any(
            isinstance(u, dict) and u.get("mode") == "on_battery"
            for u in power.values()
        )
        if on_battery:
            self.time_on_battery_s += dt_s
        if power.get("utility_available") is False and self._utility_lost_at_s is None:
            self._utility_lost_at_s = now
        if (
            power.get("on_generator") is True
            and self._generator_online_at_s is None
            and self._utility_lost_at_s is not None
        ):
            self._generator_online_at_s = now

    # ----------------------------------------------------------------------
    @property
    def mean_pue(self) -> Optional[float]:
        if self._pue_time_total_s <= 0.0:
            return None
        return self._pue_time_weighted / self._pue_time_total_s

    @property
    def generator_online_latency_s(self) -> Optional[float]:
        if self._utility_lost_at_s is None or self._generator_online_at_s is None:
            return None
        return max(0.0, self._generator_online_at_s - self._utility_lost_at_s)

    @property
    def invalid_command_rate(self) -> float:
        return self.n_invalid_command / self.n_commands if self.n_commands else 0.0

    @property
    def malformed_target_rate(self) -> float:
        return self.n_malformed_target / self.n_commands if self.n_commands else 0.0

    @property
    def invalid_arg_rate(self) -> float:
        return self.n_invalid_arg / self.n_commands if self.n_commands else 0.0

    def summary(self) -> dict:
        """Return a JSON-serialisable dict of the engineer-facing outcomes."""
        return {
            "peak_inlet_c": round(self.peak_inlet_c, 3),
            "degree_min_over_recommended": round(self.degree_min_over_recommended, 3),
            "degree_min_over_allowable": round(self.degree_min_over_allowable, 3),
            "min_ups_soc": (
                round(self.min_ups_soc, 4) if self.min_ups_soc is not None else None
            ),
            "time_on_battery_s": round(self.time_on_battery_s, 1),
            "generator_online_latency_s": (
                round(self.generator_online_latency_s, 1)
                if self.generator_online_latency_s is not None
                else None
            ),
            "mean_pue": round(self.mean_pue, 4) if self.mean_pue is not None else None,
            "total_energy_kwh": round(self.total_energy_kwh, 4),
            "invalid_command_rate": round(self.invalid_command_rate, 4),
            "malformed_target_rate": round(self.malformed_target_rate, 4),
            "invalid_arg_rate": round(self.invalid_arg_rate, 4),
            "n_commands": self.n_commands,
        }
