---
id: 9a4920a235
question: 'Project: Render deployed the wrong service from my multi-stage Dockerfile — why?'
sort_order: 29
---

Render builds your Dockerfile with no `--target` flag, so a target-less `docker build` always builds whichever stage is declared last in the file.

If your Dockerfile has multiple stages (for example, one for an API and one for a UI, sharing a common base layer), whichever stage comes last in the file is the one Render's Blueprint deploy actually runs — Render's spec has no field to pick a target stage, so it always takes the default. Reordering stages, or adding a new stage at the end, silently changes what gets deployed with no error at build time; the build succeeds either way, it just ships the wrong image. `docker compose build` is not affected by this the same way, since compose services set `target:` explicitly per service — so a Dockerfile can look correct in local `docker compose` testing while still deploying wrong on Render.

- Put the service you want Render to deploy in the *last* stage of the Dockerfile.
- Add a comment directly above that stage warning future edits not to append anything below it.
- If you add a new service/stage later, put it earlier in the file, not at the end.
- Verify what actually deployed by hitting a route unique to that service, not just a shared `/health` check both services might answer identically.
