"""Narrow reusable-resource port for the trigger domain owner."""

from weakref import ref


class SequenceTriggerResourceLifecyclePort:
    """Expose TCP resource operations without exposing domain teardown."""

    def __init__(self, target) -> None:
        try:
            self._target_ref = ref(target)
            self._strong_target = None
        except TypeError:
            self._target_ref = None
            self._strong_target = target

    def _target(self):
        if self._target_ref is not None:
            return self._target_ref()
        return self._strong_target

    @property
    def model(self):
        target = self._target()
        return None if target is None else target.model

    @property
    def _lifecycle_lock(self):
        target = self._target()
        return None if target is None else target._lifecycle_lock

    @property
    def _lifecycle_state(self):
        target = self._target()
        return "INACTIVE" if target is None else target._lifecycle_state

    @property
    def _tcp_stop_completed_handles(self):
        target = self._target()
        return {} if target is None else target._tcp_stop_completed_handles

    @property
    def _tcp_stop_journal(self):
        target = self._target()
        return None if target is None else target._tcp_stop_journal

    @property
    def _resource_identity_epoch(self):
        target = self._target()
        return 0 if target is None else target._resource_identity_epoch

    @_resource_identity_epoch.setter
    def _resource_identity_epoch(self, value) -> None:
        target = self._target()
        if target is not None:
            target._resource_identity_epoch = value

    def _admit_canonical_tcp_mirror_identity_locked(
        self, previous, current
    ) -> bool:
        target = self._target()
        if target is None:
            return current is None
        return target._admit_canonical_tcp_mirror_identity_locked(
            previous, current
        )

    def stop_tcp(self):
        target = self._target()
        return True if target is None else target.stop_tcp()

    def set_tcp_enabled(self, enabled, **options):
        target = self._target()
        if target is None:
            return False
        return target.set_tcp_enabled(enabled, **options)
