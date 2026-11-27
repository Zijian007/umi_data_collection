"""Canonical arm validation and declared-subset inheritance; no file or device IO."""

ARMS = ('left', 'right')


def read_arms(value, available=ARMS, *, label='预期使用的手') -> tuple[str, ...]:
    """Validate a nonempty subset and return it in stable left/right order."""
    if (not isinstance(value, (list, tuple)) or not value
            or any(not isinstance(arm, str) or arm not in ARMS for arm in value)
            or len(set(value)) != len(value) or set(value) - set(available)):
        raise ValueError(f'{label}：预期使用的手必须为 left、right 或两者，且属于可用手')
    return tuple(arm for arm in ARMS if arm in value)


def resolve_arms(declaration: dict, available) -> tuple[str, ...]:
    """Missing declaration inherits the available arms; explicit null is invalid.

    Tasks inherit the batch default; legacy batch snapshots inherit their frozen
    kit. The public collection schema requires arms explicitly.
    """
    return read_arms(declaration.get('expected_arms', tuple(available)), available)
