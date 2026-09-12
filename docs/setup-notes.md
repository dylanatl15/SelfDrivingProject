# Setup notes

## Normal path

```bash
uv sync --extra dev
uv run pytest
```

## When `uv sync` crawls under WSL2

On the development machine, downloads from inside WSL2 ran at ~0.1 MB/s while Windows,
on the same connection, did ~18 MB/s. If yours behaves the same, download on the Windows
side and install from a local folder. Windows' `curl.exe` is callable straight from WSL.

1. **Find exact filenames.** For PyPI packages, `https://pypi.org/pypi/<name>/<version>/json`
   lists every file under `urls`. Torch CPU wheels are listed at
   `https://download.pytorch.org/whl/cpu/torch/`. The Linux wheel is
   `torch-2.13.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl`. A guessed `linux_x86_64`
   name returns **403**, not 404, which looks like an access problem but is a typo.

2. **Download on Windows** into a shared folder:

   ```bash
   curl.exe -L -o 'C:\Users\Public\wheelhouse\<file>' '<url>'
   ```

3. **Install offline** from WSL:

   ```bash
   uv venv --python 3.12
   uv pip install --no-index --find-links /mnt/c/Users/Public/wheelhouse <packages>
   ```

4. **The editable install of this package** needs its build backend already present,
   because `--no-index` stops build isolation from fetching it. Put `hatchling`,
   `editables` and `tomlkit` (plus hatchling's own dependencies) in the wheelhouse, then:

   ```bash
   uv pip install --no-index --find-links /mnt/c/Users/Public/wheelhouse --no-build-isolation -e '.[dev]'
   ```

Wheels for other platforms can be skipped by excluding on the platform tag
(`win_amd64`, `macosx`, `musllinux`, `aarch64`), not by requiring `cp312`, which would
also drop pure-Python `py3-none-any` wheels and platform wheels such as `ruff`.
