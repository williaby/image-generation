# Response-Aware Development (RAD)

RAD markers flag assumptions that could cause production failures if wrong.
They are paired with verification instructions so future maintainers know
what to check before changing the code.

## Marker syntax

```python
# #CRITICAL -- <what breaks if this assumption is wrong>
# #ASSUME   -- <the assumption being made>
# #EDGE     -- <the edge case that could silently fail>
# #VERIFY   -- <actionable verification step>
```

## Mandatory RAD categories for this project

| Category | Location | Required markers |
|---|---|---|
| External API (Gemini) | `generate_image()`, `generate_story_sequence()` | `#ASSUME` on response schema, `#EDGE` on empty candidates, `#CRITICAL` on API key |
| External API (Topaz) | `topaz_enhance_image()` | Already annotated |
| File I/O | `document_image_prompt()`, `load_image_as_base64()` | `#EDGE` on missing/unreadable files |
| Security | Any code handling API keys or user-supplied paths | `#CRITICAL` |

## Current annotation status

### `topaz_enhance_image()` -- annotated

- Line 280: `#CRITICAL` -- API key present in Authorization header; do not log
  request headers in debug mode.
- Line 281: `#ASSUME` -- Topaz API accepts `multipart/form-data` with
  `data=` + `files=`. Verify: check Topaz changelog at
  `developer.topazlabs.com` before upgrading `requests`.
- Line 324: `#ASSUME` -- 25 polling iterations x max 12s backoff = ~5 min max
  wall time. Verify: confirm against Topaz SLA docs.
- Line 325: `#EDGE` -- 429 exhaustion without `Retry-After` header causes
  silent timeout. Verify: test with mock that never returns non-429.
- Line 386: `#ASSUME` -- download URL hostname is always in `_TOPAZ_DOWNLOAD_HOSTS`.
  Verify: check Topaz CDN policy if they announce infrastructure changes.

### `generate_image()` -- not yet annotated (P1 gap)

See `scripts/generate_image.py` around lines 667 and 729. RAD markers to add:
- `#CRITICAL` -- `api_key` is passed to `genai.Client()`; ensure it is never
  logged.
- `#ASSUME` -- `response.candidates[0].content.parts[0].inline_data.data` is
  the path to the image bytes. Verify against google-genai SDK changelog.
- `#EDGE` -- `candidates` list may be empty if Gemini refuses the prompt
  (safety filter). Currently causes `IndexError`. Add guard.

## Verification workflow

1. Before changing any annotated function, read its `#ASSUME` and `#VERIFY`
   comments.
2. Validate the assumption still holds (check API docs, run integration test).
3. Update the marker if the assumption changed.
4. Add a `#VERIFY` result comment with date: `# VERIFIED 2026-04-18: confirmed`.
