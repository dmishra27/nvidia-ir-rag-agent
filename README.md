# dvc-storage

This branch is a DVC HTTP remote for
[`nvidia-ir-rag-agent`](https://github.com/dmishra27/nvidia-ir-rag-agent),
served over `raw.githubusercontent.com`. It holds nothing but content-addressed
blobs under `files/md5/<hash[:2]>/<hash[2:]>` — the exact layout DVC's cache
uses locally — so `dvc pull` on `main` can fetch them with no cloud account,
no credentials, and no setup beyond cloning the repo.

**Do not edit or rebase this branch by hand.** It is regenerated from
`.dvc/cache` whenever the DVC-tracked files under `data/raw/` on `main`
change; treat it as a build artifact, not source. See `.dvc/config` on
`main` for the remote configuration, and `docs/uat/correction_notice_a1.md`
§6 (DEF-19/DEF-20) for why this exists.
