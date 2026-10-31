"""Durable cohort qualification and separate pre/normal validation clocks."""

from copy import deepcopy

from pvz_rl.learning.budget import validation_interval_games


class ValidationTrigger:
    def __init__(self, cfg, state=None):
        settings = cfg["training"].get("validation_trigger", {})
        self.enabled = settings.get("enabled", False)
        self.threshold = settings.get("cohort_success_threshold", 0.9)
        self.intervals = {
            domain: validation_interval_games(cfg, domain)
            for domain in ("normal", "pre", "peashooter", "cherry_bomb")
        }
        self.state = deepcopy(state) if state else dict(last_check={}, decision=None, pending=None)

    def interval_for(self, domain):
        return self.intervals[domain]

    @property
    def pending(self):
        return self.state["pending"] if self.enabled else None

    def request(self, kind, domain, progress, cohort):
        """Return (allowed, newly_checked); retries keep the original qualification."""
        if not self.enabled:
            return True, False
        previous = self.state["decision"]
        same = (
            previous is not None and previous["domain"] == domain and previous["cohort"] == cohort
        )
        if self.pending:
            if not same:
                raise ValueError("Pending validation weights or cohort changed")
            return self.pending == kind, False
        if same:
            allowed = previous["eligible"] and kind not in previous["completed"]
            if allowed:
                self.state["pending"] = kind
            return allowed, False
        last = self.state["last_check"].get(domain, 0)
        if progress - last < self.interval_for(domain):
            return False, False
        full = (
            cohort.get("games", 0) == cohort.get("configured_games") and cohort.get("games", 0) > 0
        )
        wins, games = cohort.get("wins", 0), cohort.get("games", 0)
        eligible = full and wins / games > self.threshold and cohort.get("domain") == domain
        self.state["last_check"][domain] = progress
        self.state["decision"] = dict(
            domain=domain,
            progress=progress,
            cohort=deepcopy(cohort),
            eligible=eligible,
            reason="qualified" if eligible else "cohort_success" if full else "short_cohort",
            completed=[],
        )
        if eligible:
            self.state["pending"] = kind
        return eligible, True

    def complete(self, kind):
        if self.enabled:
            if self.pending != kind:
                raise ValueError("Validation completion has no matching pending request")
            self.state["decision"]["completed"].append(kind)
            self.state["pending"] = None

    def snapshot(self, domain):
        return dict(
            enabled=self.enabled,
            cohort_success_threshold=self.threshold,
            interval_episodes=self.interval_for(domain),
            next_check=self.state["last_check"].get(domain, 0) + self.interval_for(domain),
            **deepcopy(self.state),
        )
