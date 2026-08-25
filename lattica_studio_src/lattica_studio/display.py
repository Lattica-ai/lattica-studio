from collections.abc import Iterable, Sequence


def display_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    empty_message: str,
) -> None:
    rendered_rows = [
        ["—" if value is None else str(value) for value in row]
        for row in rows
    ]
    if not rendered_rows:
        print(empty_message)
        return

    widths = [
        max(len(str(header)), *(len(row[index]) for row in rendered_rows))
        for index, header in enumerate(headers)
    ]

    def render(row: Sequence[object]) -> str:
        cells = [str(value).ljust(width) for value, width in zip(row, widths)]
        cells[-1] = cells[-1].rstrip()
        return "  ".join(cells)

    print(render(headers))
    print(render(tuple("─" * width for width in widths)))
    for row in rendered_rows:
        print(render(row))
