# Zebrafish Fish1 results

**Verdict: BLOCKED — no module, no scores.**

## What blocked access

1. **CAVE authentication required.**  
   `caveclient.CAVEclient("fish1_v250915")` with no token raises:
   `AuthException: You have not setup a token to access https://global.daf-apis.com/info/api/v2/datastack/full/fish1_v250915`  
   Token setup docs: https://caveconnectome.github.io/CAVEclient/tutorials/authentication/  
   Token mint URL used by companion sites: https://global.brain-wire-test.org/sticky_auth/settings/tokens (Google account). Editing privileges may also need the Fish1 access form (see release Proofreading page).

2. **No public flat edge export found.**  
   Release / VAST pages expose EM + segmentation (`.vsvi`), not a downloadable `pre,post,weight` synapse table. `zebrafishconnectome.vercel.app/data` reports Fish1 materialization/connectivity **unavailable** without `CAVE_TOKEN`.

## What was not done

- No Fish1 edges downloaded.
- No `fish_module.npz` built.
- No FISH vs ER/SHUF/DENS runs.
- No FRANKEN+FISH graft.
- No myelin REAL vs SHUF on fish tissue.
- No synthetic zebrafish graph invented (per project rules).

## Unlock path (for the user)

1. Create a CAVE / DAF API token with a Google account.  
2. Put it only in a local env var (e.g. `CAVE_TOKEN` / vault) — never chat, never git.  
3. Re-run the build in `PREREGISTRATION_zebrafish.md`.

## Side note (zebra finch)

Finch Area X **does** have a public flat synapse dump (j0251). Module already at `regions/finch/data/areax_module.npz` (N=7576, nnz≈245k). That experiment is separate and does not substitute for Fish1.

See `SOURCE_fish1.json` for machine-readable block details.
