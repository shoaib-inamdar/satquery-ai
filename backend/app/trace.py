"""
trace.py — Timed execution trace for the agentic pipeline.
"""
from __future__ import annotations
import time
import datetime
from typing import Any
from .schemas import TraceStage, ExecutionTrace


class Tracer:
    def __init__(self, session_id: str, task: str, router: str,
                 tool_chain: list[str], params: dict):
        self.session_id = session_id
        self.task = task
        self.router = router
        self.tool_chain = tool_chain
        self.params = params
        self.warnings: list[str] = []
        self._stages: list[TraceStage] = []
        self._current: TraceStage | None = None

    def start_stage(self, stage: str, **details: Any) -> None:
        now = datetime.datetime.utcnow().isoformat() + "Z"
        self._current = TraceStage(stage=stage, started_at=now, details=dict(details))
        self._t0 = time.perf_counter()

    def end_stage(self, **details: Any) -> None:
        if self._current:
            elapsed_ms = (time.perf_counter() - self._t0) * 1000
            self._current.ended_at = datetime.datetime.utcnow().isoformat() + "Z"
            self._current.duration_ms = round(elapsed_ms, 1)
            self._current.details.update(details)
            self._stages.append(self._current)
            self._current = None

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def build(self) -> ExecutionTrace:
        return ExecutionTrace(
            session_id=self.session_id,
            task=self.task,
            router=self.router,
            tool_chain=self.tool_chain,
            params=self.params,
            warnings=self.warnings,
            stages=self._stages,
        )
