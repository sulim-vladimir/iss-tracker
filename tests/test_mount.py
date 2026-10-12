

def test_truncated_position_reply_is_not_accepted():
    """USB noise delivered "P 1234" once; parsing it killed the keepalive thread."""
    from issctl.mount import _valid_p

    assert _valid_p("P 1234 -567")
    assert _valid_p("P 1234 -567 1")
    for bad in ("P 1234", "P", "P 12x4 5", "P nan 5", "PX 1 2"):
        assert not _valid_p(bad), bad


def test_only_one_program_drives_the_mount(tmp_path):
    """A second claim fails before anything opens the port (opening it resets the Uno)."""
    import pytest
    from issctl.mount import claim_mount
    first = claim_mount(tmp_path / "mount.lock")
    with pytest.raises(RuntimeError, match="in use by pid"):
        claim_mount(tmp_path / "mount.lock")
    first.close()                              # the program ends: the mount is free again
    claim_mount(tmp_path / "mount.lock").close()
