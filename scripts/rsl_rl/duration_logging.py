"""RSL-RL console durations that retain days for long tactile training runs."""

from contextlib import redirect_stdout
import io
import re
import sys

from rsl_rl.runners import OnPolicyRunner


def format_duration(seconds: float) -> str:
    days, remainder = divmod(max(0, int(seconds)), 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{days}d {clock}" if days else clock


class DurationLoggingOnPolicyRunner(OnPolicyRunner):
    """Keep upstream metrics and logging, correcting only console durations."""

    def log(self, locs: dict, width: int = 80, pad: int = 35, **kwargs):
        output = io.StringIO()
        with redirect_stdout(output):
            # Server RSL-RL forks pass print_terminal=False on nonzero ranks
            # while still writing their metrics. Preserve the upstream options.
            super().log(locs, width=width, pad=pad, **kwargs)
        completed = locs["it"] - locs["start_iter"] + 1
        remaining = max(0, locs["num_learning_iterations"] - completed)
        durations = {
            "Time elapsed": self.tot_time,
            "ETA": self.tot_time / completed * remaining,
        }
        rendered = output.getvalue()
        for label, seconds in durations.items():
            rendered = re.sub(
                rf"(?m)^([ \t]*{re.escape(label)}:)[^\n]*",
                lambda match: f"{match.group(1)} {format_duration(seconds)}",
                rendered,
            )
        sys.stdout.write(rendered)
        sys.stdout.flush()
