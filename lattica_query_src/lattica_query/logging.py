import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Self, TextIO

_stdout_lock = threading.Lock()
_RESET = "\033[0m"
_ERROR = "\033[38;5;196m"


@dataclass(frozen=True, slots=True)
class OutputConfig:
    """Controls terminal output in the current execution context.

    ``color`` and ``animate`` default to terminal detection. Setting
    ``enabled`` to false suppresses status and operation output entirely.
    """

    enabled: bool = True
    color: bool | None = None
    animate: bool | None = None


_output_config: ContextVar[OutputConfig | None] = ContextVar(
    "lattica_output_config",
    default=None,
)
_operation_stack: ContextVar[tuple["OperationLog", ...]] = ContextVar(
    "lattica_operation_stack",
    default=(),
)


@contextmanager
def output_context(config: OutputConfig) -> Iterator[None]:
    """Apply output settings to operations executed inside this context."""
    if not isinstance(config, OutputConfig):
        raise TypeError("config must be an OutputConfig")
    token = _output_config.set(config)
    try:
        yield
    finally:
        _output_config.reset(token)


def current_animation() -> "OperationLog | None":
    stack = _operation_stack.get()
    return stack[-1] if stack else None


def _current_output_config() -> OutputConfig:
    return _output_config.get() or OutputConfig()


def log_status(message: str) -> None:
    """Report ephemeral progress for the current operation."""
    operation = current_animation()
    if operation is not None:
        operation.set_detail(message)
    elif _current_output_config().enabled:
        print(message)


def log_info(message: str) -> None:
    """Report persistent information for the current operation."""
    operation = current_animation()
    if operation is not None:
        operation.add_info(message)
    elif _current_output_config().enabled:
        print(message)


def log_size_info(name: str, file_size: int) -> None:
    if file_size >= 1024**2:
        size_text = f"{file_size / 1024**2:.1f} MB"
    elif file_size >= 1024:
        size_text = f"{file_size / 1024:.1f} KB"
    else:
        size_text = f"{file_size} B"
    log_info(f"{name} size: {size_text}")


@dataclass(frozen=True, slots=True)
class LatticaTheme:
    label: str
    primary: str
    bright: str
    white: str
    gray: str
    dark_gray: str
    wave_colors: tuple[str, ...]


QUERY_THEME = LatticaTheme(
    label="LATTICA QUERY",
    primary="\033[38;5;39m",
    bright="\033[38;5;45m",
    white="\033[97m",
    gray="\033[38;5;244m",
    dark_gray="\033[38;5;238m",
    wave_colors=(
        "\033[38;5;238m", "\033[38;5;244m", "\033[38;5;39m",
        "\033[38;5;39m", "\033[38;5;45m", "\033[97m",
        "\033[38;5;45m", "\033[38;5;39m", "\033[38;5;39m",
        "\033[38;5;244m", "\033[38;5;238m",
    ),
)

STUDIO_THEME = LatticaTheme(
    label="LATTICA STUDIO",
    primary="\033[38;5;99m",
    bright="\033[38;5;141m",
    white="\033[97m",
    gray="\033[38;5;244m",
    dark_gray="\033[38;5;238m",
    wave_colors=(
        "\033[38;5;238m", "\033[38;5;244m", "\033[38;5;99m",
        "\033[38;5;99m", "\033[38;5;141m", "\033[97m",
        "\033[38;5;141m", "\033[38;5;99m", "\033[38;5;99m",
        "\033[38;5;244m", "\033[38;5;238m",
    ),
)


class OperationLog:
    """Render one operation and collect status emitted by nested work."""

    def __init__(
        self,
        text: str,
        theme: LatticaTheme,
        width: int = 30,
        refresh_rate: float = 0.08,
    ) -> None:
        self.text = text
        self.theme = theme
        self.width = width
        self.refresh_rate = refresh_rate
        self._detail: str | None = None
        self._detail_lock = threading.Lock()
        self._info_messages: list[str] = []
        self._info_message_set: set[str] = set()
        self._info_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float | None = None
        self._parent: OperationLog | None = None
        self._stack_token: Token[tuple[OperationLog, ...]] | None = None
        self._config = OutputConfig()
        self._stream: TextIO | None = None
        self._color = False
        self._started = False

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes = int(seconds // 60)
        return f"{minutes}m {seconds - minutes * 60:.1f}s"

    def _root(self) -> "OperationLog":
        operation = self
        while operation._parent is not None:
            operation = operation._parent
        return operation

    def set_detail(self, detail: str | None) -> None:
        root = self._root()
        with root._detail_lock:
            root._detail = detail

    def _get_detail(self) -> str | None:
        with self._detail_lock:
            return self._detail

    def add_info(self, message: str) -> None:
        root = self._root()
        message = str(message)
        with root._info_lock:
            if message in root._info_message_set:
                return
            root._info_message_set.add(message)
            root._info_messages.append(message)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._config = _current_output_config()
        self._stream = sys.stdout
        self._start_time = time.perf_counter()
        stack = _operation_stack.get()
        self._parent = stack[-1] if stack else None
        self._stack_token = _operation_stack.set((*stack, self))

        if not self._config.enabled or self._parent is not None:
            return
        is_tty = bool(getattr(self._stream, "isatty", lambda: False)())
        self._color = self._config.color if self._config.color is not None else is_tty
        animate = self._config.animate if self._config.animate is not None else is_tty
        if animate:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self, *, error: BaseException | None = None) -> None:
        if not self._started:
            return
        elapsed = time.perf_counter() - self._start_time if self._start_time is not None else 0.0
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()

        if self._config.enabled:
            if self._parent is not None:
                self._parent.add_info(f"{self.text}: {self._format_duration(elapsed)}")
                if error is not None:
                    self._parent.add_info(self._error_message(error))
                self._parent.set_detail(None)
            else:
                self._render_completion(elapsed, error)

        if self._stack_token is not None:
            _operation_stack.reset(self._stack_token)
        self._thread = None
        self._start_time = None
        self._stack_token = None
        self._started = False

    @staticmethod
    def _error_message(error: BaseException) -> str:
        raw = str(error).strip()
        message = raw.splitlines()[0] if raw else type(error).__name__
        if len(message) > 240:
            message = message[:237] + "..."
        return f"error: {type(error).__name__}: {message}"

    def _render_completion(self, elapsed: float, error: BaseException | None) -> None:
        assert self._stream is not None
        marker = "✓" if error is None else "✗"
        with _stdout_lock:
            if self._color:
                marker_color = self.theme.gray if error is None else _ERROR
                self._stream.write(
                    f"\r\033[2K{self.theme.dark_gray}[{_RESET}"
                    f"{self.theme.primary}{'━' * self.width}{_RESET}"
                    f"{self.theme.dark_gray}] {self.theme.primary}{self.theme.label:<15}{_RESET}"
                    f"{self.theme.dark_gray} │ {_RESET}{self.theme.white}{self.text}{_RESET} "
                    f"{marker_color}{marker}{_RESET}\n"
                )
            else:
                self._stream.write(f"[{self.theme.label}] {self.text} {marker}\n")

            messages = [f"duration: {self._format_duration(elapsed)}", *self._info_messages]
            if error is not None:
                messages.append(self._error_message(error))
            for message in messages:
                for index, line in enumerate(str(message).splitlines()):
                    prefix = "  └─ " if index == 0 else "     "
                    if self._color:
                        color = _ERROR if message.startswith("error: ") else self.theme.gray
                        self._stream.write(
                            f"{self.theme.dark_gray}{prefix}{_RESET}{color}{line}{_RESET}\n"
                        )
                    else:
                        self._stream.write(f"{prefix}{line}\n")
            self._stream.flush()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop(error=exc_val)

    def _run(self) -> None:
        assert self._stream is not None
        wave_chars = "░░▒▓███▓▒░░"
        start_time = time.perf_counter()
        while not self._stop_event.is_set():
            frame_idx = int((time.perf_counter() - start_time) / self.refresh_rate)
            offset = frame_idx % (self.width + len(wave_chars)) - len(wave_chars)
            left = max(offset, 0)
            right = min(offset + len(wave_chars), self.width)
            parts = ["\r", self.theme.dark_gray, "["]
            if left > 0:
                parts.append("·" * left)
            if right > left:
                wave_start = max(0, -offset)
                current_color = None
                for index in range(wave_start, wave_start + right - left):
                    color = self.theme.wave_colors[index]
                    if color != current_color:
                        parts.append(color)
                        current_color = color
                    parts.append(wave_chars[index])
                parts.append(self.theme.dark_gray)
            if right < self.width:
                parts.append("·" * (self.width - right))
            parts.extend([
                "] ", self.theme.primary, f"{self.theme.label:<15}", self.theme.dark_gray,
                " │ ", self.theme.gray, self.text,
            ])
            detail = self._get_detail()
            if detail:
                parts.extend([self.theme.dark_gray, " • ", self.theme.gray, detail])
            parts.extend([_RESET, "\033[K"])
            with _stdout_lock:
                self._stream.write("".join(parts))
                self._stream.flush()
            self._stop_event.wait(self.refresh_rate)
