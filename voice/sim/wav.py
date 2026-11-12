"""Chapter 2's WAV reader, for later chapters: plain PCM and G.711 mu-law, numpy only."""

import struct
from pathlib import Path

import numpy as np


def read_wav(path):
    """Parse a WAV file by hand: a few labelled chunks of header, then the raw sample bytes."""
    data = Path(path).read_bytes()
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE", "not a WAV file"
    pos, info = 12, {"bytes": len(data)}
    while pos + 8 <= len(data):
        chunk, size = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        if chunk == b"fmt ":
            code, channels, sr, byte_rate, _, bits = struct.unpack("<HHIIHH", body[:16])
            info.update(code=code, channels=channels, sr=sr, byte_rate=byte_rate, bits=bits)
        elif chunk == b"data":
            info["raw"] = body
        pos += 8 + size + (size & 1)
    return info


def mulaw_decode(raw):
    """G.711 mu-law: each byte is a sign bit, a 3-bit exponent and a 4-bit mantissa, stored inverted."""
    u = ~np.frombuffer(raw, dtype=np.uint8)
    exponent, mantissa = (u >> 4) & 7, (u & 15).astype(np.int32)
    magnitude = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return np.where(u & 0x80, -magnitude, magnitude) / 32768


def load(path):
    """Samples as float32 in [-1, 1], and the sample rate."""
    w = read_wav(path)
    if w["code"] == 1 and w["bits"] == 16:
        x = np.frombuffer(w["raw"], dtype="<i2") / 32768
    elif w["code"] == 7:
        x = mulaw_decode(w["raw"])
    else:
        raise ValueError(f"WAV format {w['code']} is not handled here")
    if w["channels"] > 1:
        x = x.reshape(-1, w["channels"]).mean(axis=1)
    return x.astype(np.float32), w["sr"]
