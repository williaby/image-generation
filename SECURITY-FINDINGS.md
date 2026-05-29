# Security Review Findings

Review scope: `scripts/generate_image.py` (the only application code in the
repository), the GitHub Actions workflows under `.github/workflows/`, and the
declared dependency surface (`pyproject.toml`).

Date: 2026-05-15
Branch reviewed: `claude/security-review-image-gen-4iVzL`

## 1. Architecture and threat model

`generate_image.py` is a **single-user command-line tool**. It:

- Reads CLI arguments via `argparse`.
- Loads API keys for Google Gemini (`GEMINI_API_KEY`) and Topaz Labs
  (`TOPAZ_API_KEY`) from environment variables, with a fallback to a
  `.env` file in the repository root.
- Calls the Google Gemini API (`google-genai` SDK) for image generation.
- Optionally calls the Topaz Labs REST API (`requests`) for post-processing.
- Writes generated images to the local `output/` directory.

There is **no HTTP server, no REST endpoint, no multi-tenant boundary, and
no upload surface**. The user who invokes the script supplies their own
prompts, their own API keys, and reads files they own. The relevant
threat model is therefore:

- Integrity of the supply chain (workflows, pinned actions, dependencies).
- Defense-in-depth against later misuse (e.g. user serves `output/` over
  HTTP, pastes generated `PROMPTS.md` into a rendered Markdown viewer, or
  pipes an attacker-controlled image through `--enhance`).
- Avoiding accidental key leakage.

Items in the requested checklist that are not applicable have been called
out explicitly below rather than skipped silently.

## 2. Findings

### 2.1 Prompt handling

| # | Item | Status |
|---|------|--------|
| 2.1.1 | API keys loaded from environment | OK |
| 2.1.2 | Prompt injection from external user input | N/A (single-user CLI) |
| 2.1.3 | Content filtering on prompts and outputs | Delegated to Gemini |

- **2.1.1 API key loading** &mdash; `_load_api_key` (`scripts/generate_image.py:174`)
  reads from `os.environ` first, then optionally from `.env` at repo
  root. `.env` is in `.gitignore`. The `genai.Client(api_key=...)` and
  Topaz `X-API-KEY` header are the only places keys leave the function;
  neither is logged. Inline `#CRITICAL` markers at
  `scripts/generate_image.py:280` and `scripts/generate_image.py:672`
  call out the contract for future maintainers.
- **2.1.2 Prompt injection** &mdash; The script is invoked by a single
  user who supplies their own prompt text. There is no second party
  whose input is incorporated into a generation call. In the multi-part
  story flow (`generate_story_sequence`, line ~970), the per-part
  prompts are hard-coded strings; the previous image is passed as a
  separate `Part`, not concatenated into the textual prompt. No
  injection vector exists in the current design. If this tool is ever
  wrapped as a service, the prompt-input boundary will need explicit
  validation.
- **2.1.3 Content filtering** &mdash; Both the Gemini and Topaz APIs apply
  their own content policies server-side. The tool does no additional
  filtering and does not need to in the current single-user model. The
  response handler at `scripts/generate_image.py:768` already inspects
  `prompt_feedback` when no candidates are returned (e.g. a model
  refusal).

### 2.2 File handling

| # | Item | Status |
|---|------|--------|
| 2.2.1 | Non-guessable default filenames | **Fixed (defense-in-depth)** |
| 2.2.2 | Path traversal on retrieval | N/A (no retrieval endpoint) |
| 2.2.3 | File size limits on input images | **Fixed** |
| 2.2.4 | Markdown injection into `PROMPTS.md` | **Fixed** |
| 2.2.5 | Topaz download URL allow-list | OK (already present) |

- **2.2.1 Non-guessable default filenames** &mdash; Previously
  default-generated files used second-resolution timestamps:
  `generated_YYYYMMDD_HHMMSS.png`. While there is no remote retrieval
  endpoint in the current code, a user could later serve `output/` via
  a static web server; predictable filenames would then enumerate
  trivially. Fix: appended an 8-character `secrets.token_hex(4)` to all
  default filenames (`generated_*`, `draft_*`, `thought*`, `story_*`).
  Test assertion `"generated_" in result.name` still passes.
- **2.2.2 Path traversal** &mdash; No HTTP-exposed retrieval. CLI users
  pass `-o`, `-r`, `--enhance`, `--finalize` paths under their own user
  account; access is bounded by OS permissions. In
  `generate_image()` (`scripts/generate_image.py:872`+) the script
  already takes only `output_path.name` when re-rooting under
  `output/`, so traversal via `-o ../../etc/foo` does not escape
  `output/`. Topaz output respects the literal user-supplied path,
  which is the documented and expected CLI behaviour.
- **2.2.3 File size limits** &mdash; Previously, `load_image_as_base64`
  (`scripts/generate_image.py:497`) and `topaz_enhance_image`
  (`scripts/generate_image.py:216`) read the entire input image into
  memory with no upper bound. Fix: added `MAX_INPUT_IMAGE_BYTES`
  (100 MiB) and `Path.stat().st_size` checks at both entry points.
  Limit is comfortably above legitimate 4K source images (~10 MiB).
- **2.2.4 Markdown injection into `PROMPTS.md`** &mdash; `document_image_prompt`
  (`scripts/generate_image.py:522`) embeds the user's prompt verbatim
  inside a triple-backtick fence and inside a Markdown table row. A
  prompt containing triple backticks would terminate the fence early,
  and a prompt containing `|` or a newline would break the table
  layout. While the user feeds their own data here, the file is meant
  to be rendered by other tools (web viewers, agent inputs) and a
  malformed entry can corrupt downstream tooling. Fix:
  - Table cell: escape `\` and `|`, strip newlines.
  - Detailed entry: render the prompt as an **indented code block**
    (6-space prefix per line: 2 spaces for list continuation plus 4
    spaces for the code block), which is invulnerable to in-prompt
    backtick fences.
- **2.2.5 Topaz download URL allow-list** &mdash; Already present at
  `scripts/generate_image.py:393`. The script restricts the download
  to HTTPS URLs whose hostname is in
  `{api.topazlabs.com, cdn.topazlabs.com}` and explicitly disables
  redirects on the download GET. This is good SSRF hardening and was
  retained.

### 2.3 API security

| # | Item | Status |
|---|------|--------|
| 2.3.1 | REST endpoint authentication | N/A (no endpoints) |
| 2.3.2 | Rate limiting against cost abuse | N/A (single-user, BYO key) |

The script does not expose any endpoints. Cost exposure is bounded by
the calling user's own Gemini / Topaz quotas and API keys. The Topaz
polling loop already has a 25-iteration cap and exponential back-off on
429 responses (`scripts/generate_image.py:330`).

### 2.4 GitHub Actions hardening

Audit against the requested checklist (pin all `uses:` to commit SHA,
add `permissions:` blocks, add `step-security/harden-runner` with
`egress-policy: audit`):

| Workflow | SHA-pinned `uses:` | `permissions:` block | `harden-runner` |
|---|---|---|---|
| `ci.yml` | yes | yes | yes (gate job) |
| `codeql.yml` | yes | yes | yes |
| `coverage.yml` | **partial &mdash; @main** | yes | (delegated) |
| `pr-validation.yml` | yes | yes | yes |
| `python-compatibility.yml` | yes (SHA) | yes | (delegated) |
| `repo-health.yml` | n/a (no `uses:`) | **added by this PR** | **added by this PR** |
| `reuse.yml` | yes | yes | yes |
| `sbom.yml` | yes | yes | (delegated) |
| `scorecard.yml` | yes | yes | (delegated) |
| `security-analysis.yml` | yes | yes | yes (gate job) |

Fixes applied in this PR:

- **`repo-health.yml`**: added top-level `permissions: contents: read`,
  a job-level `permissions: contents: read`, and a `harden-runner` step
  with `egress-policy: audit`.

Resolved in a follow-up commit on this branch:

- **`coverage.yml` line 26**: was previously pinned to `@main` (floating
  ref). Now pinned to commit SHA
  `732c0e313c250b7702d1a9ba75bfc4edb07dd830` of
  `ByronWilliamsCPA/.github`. Renovate will surface upstream updates.
- **`repo-health.yml` `harden-runner`**:
  initially introduced at v2.10.1 to match `codeql.yml` /
  `security-analysis.yml` / `pr-validation.yml`. Now upgraded to v2.19.1
  (`a5ad31d6a139d249332a2605b85202e8c0b78450`) to match `ci.yml` and
  `reuse.yml`, eliminating the within-repo version skew. The same upgrade
  should be applied to the other three callers in a separate PR.

Subsequent change (recorded for posterity):

- **`fips-compatibility.yml` was removed entirely in PR #29** (commit
  `d6752fd`) after WF-16 deployment surfaced repeated `FileNotFoundError`
  failures: this repo does not contain `scripts/check_fips_compatibility.py`
  and is therefore out of scope for the WF-16 FIPS check policy. Any
  hardening guidance that previously targeted that workflow no longer
  applies here.

Outstanding (recommend follow-up, not auto-applied):

- **Reusable-workflow caller jobs** (`coverage.yml`, `python-compatibility.yml`,
  `sbom.yml`, `scorecard.yml`): a caller job whose only step is `uses:`
  cannot add a sibling `harden-runner` step. Egress hardening for
  these jobs must live inside the reusable workflow itself. This is
  not a regression; the org reusable workflow is the right place to
  enforce it.

- **`harden-runner` egress policy is `audit` (log-only)**: blocking mode
  requires curating an explicit allow-list per workflow. Suggested
  follow-up: enumerate egress destinations for the repo-health job and
  switch it to `block`.

### 2.5 Dependency surface

Declared runtime dependencies:

- `google-genai>=0.4.0,<2.0.0`
- `requests>=2.32.0,<3.0.0`

Both are bounded by upper version ceilings. `docs/known-vulnerabilities.md`
records a clean `pip-audit` as of 2026-04-18 with the next review due
2026-06-17.

Transitive dependency update relevant to this review:

- `urllib3` 2.6.3 &rarr; 2.7.0 to clear two CVEs disclosed against the
  prior pin: CVE-2026-44431 (cross-origin sensitive-header forwarding in
  proxied low-level redirects; affects 1.23..&lt;2.7.0) and CVE-2026-44432
  (decompression-bomb safeguards bypassed on parts of the streaming API;
  affects 2.6.0..&lt;2.7.0). The pin and `uv.lock` regeneration were
  applied in **PR #24** (commit `8a9d79c`), which added an explicit
  `urllib3>=2.7.0,<3.0.0` entry to `[project].dependencies`; recorded
  here for the security posture record. See
  `docs/known-vulnerabilities.md` for the closed-CVE entry.

## 3. Summary of changes in this PR

Code:

- `scripts/generate_image.py`
  - Add `MAX_INPUT_IMAGE_BYTES` and enforce it in `load_image_as_base64`
    and `topaz_enhance_image`.
  - Append a `secrets.token_hex(4)` token to default-generated
    filenames in `generate_image`, `generate_story_sequence`, and the
    thought-image sidecar.
  - Sanitize Markdown table and switch to indented code blocks in
    `document_image_prompt` so prompt content cannot break out of the
    generated `PROMPTS.md`.

Workflows:

- `.github/workflows/repo-health.yml`: add `permissions:` blocks and a
  `harden-runner` step.

Dependencies (applied in PR #24, recorded here):

- `urllib3` pinned to `>=2.7.0,<3.0.0` in `pyproject.toml`; `uv.lock`
  regenerated (`urllib3` 2.6.3 &rarr; 2.7.0) to close CVE-2026-44431 and
  CVE-2026-44432. See `docs/known-vulnerabilities.md` for the closed-CVE
  entry.

Documentation:

- This file (`SECURITY-FINDINGS.md`).

Test status: 149/149 tests pass with 90.50% coverage
(`uv run --frozen --extra dev pytest tests/`).
`uv run ruff check scripts/` reports no issues.
