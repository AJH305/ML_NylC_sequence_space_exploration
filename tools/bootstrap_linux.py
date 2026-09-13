"""Install pinned uv into this project only, without modifying system Python."""

import io
import json
from pathlib import Path
import platform
import urllib.request
import zipfile

root = Path(__file__).resolve().parents[1]
target = root / ".tools-linux/uv"
if not target.exists():
    machine = {"x86_64": "x86_64", "aarch64": "aarch64"}[platform.machine()]
    with urllib.request.urlopen("https://pypi.org/pypi/uv/0.12.13/json") as response:
        metadata = json.load(response)
    wheel = next(
        f
        for f in metadata["urls"]
        if f["filename"].endswith(".whl") and "manylinux_2_17_" + machine in f["filename"]
    )
    with urllib.request.urlopen(wheel["url"]) as response:
        content = response.read()
    from hashlib import sha256

    if sha256(content).hexdigest() != wheel["digests"]["sha256"]:
        raise ValueError("uv wheel checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        member = next(n for n in archive.namelist() if n.endswith("/uv"))
        target.parent.mkdir(exist_ok=True)
        target.write_bytes(archive.read(member))
    target.chmod(0o755)
print(target)
