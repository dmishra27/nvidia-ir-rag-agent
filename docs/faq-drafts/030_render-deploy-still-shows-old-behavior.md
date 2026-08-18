---
id: dba1b8929a
question: 'Project: I pushed a fix and Render says it deployed, but the live service still behaves like the old code — what happened?'
sort_order: 30
---

Don't trust a healthcheck passing as proof your fix is live — re-verify the actual behavior you changed, against the actual deployed URL, after the push.

A green healthcheck only proves *some* container is answering `/health`; it doesn't prove it's running the image you just built. The most common cause is the build cache: your new commit really did deploy, but a Docker layer got reused from the cached build instead of rebuilt from your latest change — so the platform genuinely built and shipped "the new commit" while still serving stale content out of a stale `COPY` layer underneath it. Less commonly, the new build hasn't finished rolling out yet, autoDeploy didn't trigger on that push the way you expected, or something upstream of your fix is failing before your changed code path is even reached. From outside the platform's dashboard, a client-side check can't distinguish these — they all look identical from `curl`.

- After any fix, re-check the specific behavior you changed against the live URL — not just that the service responds at all.
- If it still looks unfixed, check the platform's own build/deploy logs before assuming your code is wrong; a working fix that never actually deployed looks identical to a broken fix.
- Don't update docs or status pages to say something is "fixed" based on the push alone — wait for a live re-check, and say "pending verification" until you have one.
- If a redeploy seems stuck, a manual "clear cache and deploy" (if your platform offers one) rules out a caching layer before you spend more time debugging application code.
