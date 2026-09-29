

def test_truncated_position_reply_is_not_accepted():
    """USB noise delivered "P 1234" once; parsing it killed the keepalive thread."""
    from issctl.mount import _valid_p

    assert _valid_p("P 1234 -567")
    assert _valid_p("P 1234 -567 1")
    for bad in ("P 1234", "P", "P 12x4 5", "P nan 5", "PX 1 2"):
        assert not _valid_p(bad), bad
