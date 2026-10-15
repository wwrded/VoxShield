"""
Download four LibriSpeech subsets of real human speech (CC BY 4.0, openslr.org).

LibriSpeech is read audiobook speech recorded by volunteers at home on their own
microphones, so it covers far more voices, rooms and microphones than the
studio recordings that make up the genuine half of ASVspoof 2019. The subsets
are speaker-disjoint, so dev-* can be used for training and test-* kept for
evaluation without any speaker appearing on both sides.

    python fetch_librispeech.py
"""
import os
import ssl
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

DEST = Path("D:/Before_Reset/Documents/SLIIT/Y3S2/ISP_Assignment/datasets/librispeech")
SUBSETS = ["dev-clean", "dev-other", "test-clean", "test-other"]
URL = "https://www.openslr.org/resources/12/{}.tar.gz"
ctx = ssl.create_default_context()
DEST.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()


def log(msg):
    with lock:
        print(msg, flush=True)


def fetch(name):
    out = DEST / f"{name}.tar.gz"
    url = URL.format(name)
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=ctx) as r:
        total = int(r.headers["Content-Length"])
    for attempt in range(1, 30):
        have = out.stat().st_size if out.exists() else 0
        if have >= total:
            break
        hdr = {"User-Agent": "Mozilla/5.0"}
        if have:
            hdr["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr),
                                        timeout=120, context=ctx) as r, \
                    open(out, "ab" if have else "wb") as fh:
                last = time.time()
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
                    if time.time() - last > 30:
                        log(f"  {name}: {out.stat().st_size / 1e6:.0f}/{total / 1e6:.0f} MB")
                        last = time.time()
        except Exception as exc:  # noqa: BLE001
            log(f"  {name}: retry {attempt} after {type(exc).__name__}")
            time.sleep(5)
    if out.stat().st_size < total:
        log(f"FAILED {name}")
        return
    if not (DEST / "LibriSpeech" / name).exists():
        with tarfile.open(out) as tf:
            tf.extractall(DEST, filter="data")
    n = sum(1 for _ in (DEST / "LibriSpeech" / name).rglob("*.flac"))
    log(f"done {name}: {total / 1e6:.0f} MB, {n} utterances")


t0 = time.time()
threads = [threading.Thread(target=fetch, args=(s,)) for s in SUBSETS]
for t in threads:
    t.start()
for t in threads:
    t.join()
log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")
sys.exit(0)
