"""Process-owned state for the single VE prewarm opportunity.

The lifetime deliberately has no Qt, worker, or SDK dependency.  A production
bootstrap owns one instance and injects it into every GUI reconstruction that
belongs to that process.
"""

from dataclasses import dataclass
from threading import RLock

from base.recording_process_protocol import _acquisition_signature


@dataclass(frozen=True)
class VePrewarmLifetimeSnapshot:
    """Immutable observation of the process-level prewarm outcome."""

    state: str
    token: object = None
    signature: object = None
    failed_signature: object = None
    failure_category: str | None = None
    code: int | None = None
    detail: str = ""
    diagnostics: tuple[str, ...] = ()
    ownership_safe: bool | None = None


class VePrewarmLifetime:
    """Thread-safe, monotonic owner of one prewarm claim per process."""

    def __init__(self):
        self._lock = RLock()
        self._snapshot = VePrewarmLifetimeSnapshot(state="available")

    @staticmethod
    def _normalize_signature(signature):
        try:
            return _acquisition_signature(signature)
        except ValueError as exc:
            raise ValueError(
                f"VE prewarm acquisition signature is invalid: {exc}") from exc

    @classmethod
    def _normalize_identity(cls, token, signature):
        if type(token) is not str or not token:
            raise ValueError("VE prewarm selection token must be a nonempty string")
        return token, cls._normalize_signature(signature)

    def snapshot(self):
        with self._lock:
            return self._snapshot

    def claim(self, token, signature):
        token, signature = self._normalize_identity(token, signature)
        with self._lock:
            if self._snapshot.state != "available":
                return False
            self._snapshot = VePrewarmLifetimeSnapshot(
                state="pending", token=token, signature=signature)
            return True

    def _matches_pending(self, token, signature):
        snapshot = self._snapshot
        return (snapshot.state == "pending"
                and snapshot.token == token
                and snapshot.signature == signature)

    def mark_succeeded(self, token, signature):
        token, signature = self._normalize_identity(token, signature)
        with self._lock:
            if not self._matches_pending(token, signature):
                return False
            self._snapshot = VePrewarmLifetimeSnapshot(
                state="succeeded",
                token=token,
                signature=signature,
                ownership_safe=True,
            )
            return True

    def mark_skipped_busy(self, token, signature, detail):
        token, signature = self._normalize_identity(token, signature)
        with self._lock:
            if not self._matches_pending(token, signature):
                return False
            if type(detail) is not str:
                raise TypeError("VE prewarm busy detail must be a string")
            self._snapshot = VePrewarmLifetimeSnapshot(
                state="skipped_busy",
                token=token,
                signature=signature,
                detail=detail,
            )
            return True

    def mark_failed(self, token, signature, fault, *, ownership_safe):
        token, signature = self._normalize_identity(token, signature)
        with self._lock:
            if not self._matches_pending(token, signature):
                return False
            if type(ownership_safe) is not bool:
                raise TypeError("VE prewarm ownership_safe must be a boolean")
            if fault is None:
                raise ValueError("VE prewarm failure requires a fault")

            category = getattr(fault, "stage", None)
            if type(category) is not str or not category:
                category = "prewarm"
            code = getattr(fault, "code", None)
            if code is not None and type(code) is not int:
                raise TypeError("VE prewarm fault code must be an integer or None")
            detail = getattr(fault, "detail", None)
            if detail is None:
                detail = str(fault)
            if type(detail) is not str or not detail:
                raise ValueError("VE prewarm fault detail is required")
            diagnostics = getattr(fault, "diagnostics", ())
            if diagnostics is None:
                diagnostics = ()
            try:
                frozen_diagnostics = tuple(diagnostics)
            except TypeError as exc:
                raise TypeError("VE prewarm diagnostics must be iterable") from exc
            if any(type(item) is not str for item in frozen_diagnostics):
                raise TypeError("VE prewarm diagnostics must contain strings")

            self._snapshot = VePrewarmLifetimeSnapshot(
                state="failed",
                token=token,
                signature=signature,
                failed_signature=signature,
                failure_category=category,
                code=code,
                detail=detail,
                diagnostics=frozen_diagnostics,
                ownership_safe=ownership_safe,
            )
            return True

    def admission_for(self, signature):
        if signature is not None:
            signature = self._normalize_signature(signature)
        with self._lock:
            snapshot = self._snapshot
            if snapshot.state == "pending":
                return "pending"
            if (snapshot.state == "failed"
                    and signature == snapshot.failed_signature):
                return "failed_signature"
            return "allowed"
