"""Small LRU registries for late duplicate suppression."""
from collections import OrderedDict


DEFAULT_TOMBSTONE_LIMIT = 512
DEFAULT_INCOMPLETE_LIMIT = 512


def as_ordered_registry(value):
    if isinstance(value, OrderedDict):
        return value
    return OrderedDict(value.items()) if isinstance(value, dict) else OrderedDict()


def touch_terminal(registry, key, value, *, limit=DEFAULT_TOMBSTONE_LIMIT):
    registry[key] = value
    registry.move_to_end(key)
    while len(registry) > int(limit):
        removable = next((candidate for candidate, record in registry.items()
                          if not isinstance(record, dict)
                          or record.get("state") != "publishing"), None)
        if removable is None:
            break
        registry.pop(removable, None)

def trim_group_registry(
        registry, *, protected_keys=(), terminal_limit=DEFAULT_TOMBSTONE_LIMIT,
        incomplete_limit=DEFAULT_INCOMPLETE_LIMIT):
    """Bound terminal tombstones and abandoned partial groups separately."""
    protected = {str(key) for key in protected_keys if str(key)}
    terminal = [key for key, value in registry.items()
                if isinstance(value, dict) and bool(value.get("terminal"))
                and str(key) not in protected]
    while len(terminal) > int(terminal_limit):
        registry.pop(terminal.pop(0), None)
    incomplete = [key for key, value in registry.items()
                  if (not isinstance(value, dict)
                      or not bool(value.get("terminal")))
                  and str(key) not in protected]
    while len(incomplete) > int(incomplete_limit):
        registry.pop(incomplete.pop(0), None)


def trim_terminal_groups(registry, *, limit=DEFAULT_TOMBSTONE_LIMIT):
    trim_group_registry(registry, terminal_limit=limit)
