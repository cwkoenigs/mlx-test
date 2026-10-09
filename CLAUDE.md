# Working in this repo

- Read `.claude/skills/mlx-cluster/SKILL.md` first; it is the workflow. The plan is `docs/MLX_DEEP_LEARNING.md`.
- Run everything through the pinned venv (`.venv/bin/python`, created by `./install.sh` / `scripts/setup_node.sh`).
  Never `pip install` into it; change `pyproject.toml`, run `uv lock`, commit, `uv sync --frozen` on both Macs.
- Distributed runs go through `scripts/launch.sh` (it passes `--cwd` and `--python` to `mlx.launch`); both Macs
  need the repo at the same absolute path and identical versions (`scripts/check_cluster.sh`).
- No pretrained models / Hugging Face downloads: the experiment builds and trains its own models on synthetic data.
- Results are deliverables: commit `results/` (small JSONL, markdown, JSON). `hosts.json` stays git-ignored.
- Before any cluster run, `./tests/selfcheck.sh` must pass on the machine you are on.
- MLX API facts verified for the pinned version live in `.claude/skills/mlx-cluster/reference/api-notes.md`;
  re-verify them if the pin changes.
- The report lives in `docs/` (MkDocs Material, `mkdocs.yml`; preview with `uvx --with mkdocs-material mkdocs serve`).
  When results change, update `results/SUMMARY.md` and the matching `docs/` page together; every number must trace to a file in `results/`.
