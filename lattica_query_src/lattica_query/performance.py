from collections.abc import Mapping

from .logging import log_info


def parse_server_timing(header: str) -> dict[str, float]:
    """
    Parses a Server-Timing string and returns exclusive durations in seconds.
    Assumes each part includes the next one.
    """
    parts: list[tuple[str, float]] = []
    for token in header.split(','):
        token = token.strip()
        if ';dur=' in token:
            timing_name, duration_text = token.split(';dur=', 1)
            try:
                parts.append((timing_name.strip(), float(duration_text) / 1000.0))
            except ValueError:
                continue

    # Compute exclusive durations
    exclusive: dict[str, float] = {}
    for i in range(len(parts) - 1):
        step_name, duration = parts[i]
        next_duration = parts[i + 1][1]
        exclusive[step_name] = duration - next_duration
    if parts:
        step_name, duration = parts[-1]
        exclusive[step_name] = duration
    return exclusive


def summarize_timing(data: Mapping[str, float], source: str) -> None:
    if not data:
        return
    grand_total = sum(data.values())
    sorted_totals = sorted(data.items(), key=lambda item: item[1], reverse=True)

    step_width = max(len("Step"), *(len(name) for name, _ in sorted_totals))
    time_width = 10
    percent_width = 8

    lines = [
        f"Timing summary • {source}",
        (
            f"{'Step':<{step_width}}  "
            f"{'Time':<{time_width}}  "
            f"{'Percent':<{percent_width}}"
        ),
        (
            f"{'─' * step_width}  "
            f"{'─' * time_width}  "
            f"{'─' * percent_width}"
        ),
    ]

    def _format_duration(seconds: float) -> str:
        if seconds < 1:
            return f"{seconds * 1000:.1f} ms"
        return f"{seconds:.3f} s"

    for name, duration in sorted_totals:
        percent = (
            duration / grand_total * 100
            if grand_total > 0
            else 0.0
        )

        duration_text = _format_duration(duration)
        percent_text = f"{percent:.1f}%"

        lines.append(
            f"{name:<{step_width}}  "
            f"{duration_text:>{time_width}}  "
            f"{percent_text:>{percent_width}}"
        )

    lines.extend([
        (
            f"{'─' * step_width}  "
            f"{'─' * time_width}  "
            f"{'─' * percent_width}"
        ),
        (
            f"{'Total':<{step_width}}  "
            f"{_format_duration(grand_total):>{time_width}}  "
            f"{'100.0%':>{percent_width}}"
        ),
    ])

    log_info("\n".join(lines))
