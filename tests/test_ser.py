import struct

import numpy as np

from issctl.ser import UNIX_TO_SER_TICKS, SerWriter

T0 = 1_700_000_000.0


def read_ser(path):
    """Minimal SER v3 reader: header, frames, then the timestamp trailer."""
    data = open(path, "rb").read()
    assert data[:14] == b"LUCAM-RECORDER"
    lu_id, color, little_endian, w, h, depth, count = struct.unpack("<7i", data[14:42])
    observer, instrument, telescope = data[42:82], data[82:122], data[122:162]
    frames_end = 178 + count * w * h * (depth // 8)
    frames = np.frombuffer(data[178:frames_end], np.uint8).reshape(count, h, w) if count else None
    stamps = struct.unpack(f"<{count}q", data[frames_end:frames_end + 8 * count])
    return {"color": color, "w": w, "h": h, "depth": depth, "count": count, "frames": frames,
             "stamps": stamps, "instrument": instrument.rstrip(b"\0").decode(),
             "telescope": telescope.rstrip(b"\0").decode(), "observer": observer.rstrip(b"\0").decode(),
             "little_endian": little_endian}


def test_ser_roundtrip(tmp_path):
    path = tmp_path / "test.ser"
    w, h, n = 64, 48, 5
    writer = SerWriter(path, w, h, bayer="RGGB", telescope="750mm", instrument="ASI290MC")
    writer.active = True
    for i in range(n):
        writer(np.full((h, w), i * 10, np.uint8), T0 + i * 0.01)
    writer.close()

    info = read_ser(path)
    assert (info["count"], info["w"], info["h"], info["depth"]) == (n, w, h, 8)
    assert info["color"] == 8            # RGGB, so stacking software debayers correctly
    assert info["instrument"] == "ASI290MC" and info["telescope"] == "750mm"
    assert info["frames"][3][0, 0] == 30  # frame order preserved
    assert list(info["stamps"]) == [int((T0 + i * 0.01 + UNIX_TO_SER_TICKS) * 1e7) for i in range(n)]
    assert writer.frames == n and writer.dropped == 0


def test_ser_records_only_while_active(tmp_path):
    path = tmp_path / "gap.ser"
    writer = SerWriter(path, 32, 32)
    writer(np.zeros((32, 32), np.uint8), T0)          # before start: ignored
    writer.active = True
    writer(np.ones((32, 32), np.uint8), T0 + 1)
    writer.active = False                              # e.g. ISS entered shadow
    writer(np.zeros((32, 32), np.uint8), T0 + 2)
    writer.close()

    info = read_ser(path)
    assert info["count"] == 1
    assert info["color"] == 0                          # mono when no Bayer pattern given
    assert info["frames"][0][0, 0] == 1


def test_ser_ignores_wrong_shape(tmp_path):
    path = tmp_path / "shape.ser"
    writer = SerWriter(path, 32, 32)
    writer.active = True
    writer(np.zeros((16, 16), np.uint8), T0)
    writer.close()
    assert read_ser(path)["count"] == 0
