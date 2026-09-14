# Human Evaluation — Online Deployment Guide

> **Superseded for the live open-link study (September 2026).** See
> [human_eval_open_protocol.md](human_eval_open_protocol.md): one shared URL,
> variable enrolment, eight pages per person, new bilingual rubric and separate
> `openv1_*` tables. Personal-link instructions below are historical; do not use
> their importer or inferential defaults for the new protocol.

This guide takes the blind rating study from `experiments/results/human_eval/...` and
puts it online so that raters anywhere can work in their own browser, and brings the
ratings back into the repository for `compute_irr.py`.

Current final protocol uses a pre-specified 30-minute burden limit. It keeps the
six-scale rubric, blind assignment and three independent ratings per output, but uses
a stratified eight-item subset of the frozen 40-item human-evaluation pool.

```text
build_human_eval.py        -> study_manifest.json, items.jsonl, assignment.json
export_human_eval_web.py   -> web/site/  (publish)   +  web/unblinding_key.json (keep private)
GitHub Pages               -> raters open a personal link, rate, progress is saved
Apps Script + Google Sheet -> every rating lands in one spreadsheet
import_human_eval_web.py   -> ratings/<rater>.jsonl  (the format the Streamlit app writes)
compute_irr.py             -> analysis/ (Krippendorff alpha, system means, paired tests)
```

Time needed once: about 20 minutes. Accounts needed: the GitHub account you already
have, plus any Google account for the collector spreadsheet.

---

## 0. What is published and what is not

| Published to the web | Never leaves the repository |
|---|---|
| Dashboard briefs (users, goals, KPIs, columns, constraints) | Which system produced which output (A/B/C/D) |
| One rendered recommendation per rating unit, addressed by an opaque token such as `u_0cc3537207ce` | Model name, seed, run identifiers, dataset digests |
| The six rubric scales with their 1/3/5 anchors | The real item identifiers |
| Rater IDs (`rater_01` … `rater_06`) | `web/unblinding_key.json`, which maps tokens back to units and methods |
| An opaque study handle such as `study_c2f1c8a054` (this is what the spreadsheet stores) | The readable study name `dashboard_v4__qwen3_8_27b__seed_42` |

The exporter refuses to write a bundle whose published payload contains a method name
or another identity marker, and every string produced by a model is HTML-escaped before
it is written, so a malformed output cannot inject markup into the page.

No personal data is collected: no name, no email address, no IP logging by the app
itself. A rater is identified only by the ID you assign.

---

## 1. Build the final study for `qwen3_8_27b`, seed 42

```bash
python experiments/scripts/build_human_eval.py --dataset dashboard_v4 --model qwen3_8_27b --seed 42 --n-items 8 --n-raters 6 --ratings-per-output 3 --item-list experiments/configs/human_eval_dashboard_v4_30min_items.csv --study-type final --planned-max-minutes 30 --estimated-minutes-per-rating 1.5 --fixed-instruction-minutes 5 --out-dir experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42_final_30min
```

This writes `experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42_final_30min/`
with 8 items × 4 methods = 32 rating units, 3 ratings per unit, 16 ratings per rater.
The workload model is 5 instruction minutes plus 1.5 minutes per rating: 29 minutes.
This is a planning estimate; actual completion time can vary.

> **Reporting note.** Directories `seed_42/` and `seed_42_final/` are superseded
> protocols. Do not pool their ratings with this final study. Report the reduced item
> count and resulting precision limitation in the thesis.

## 2. Create the collector spreadsheet

1. Open <https://sheets.new> and name the spreadsheet, for example
   `human-eval-dashboard-v4-qwen3-27b`.
2. Choose **Extensions → Apps Script**. An editor opens with an empty `Code.gs`.
3. Delete the placeholder content and paste the whole file
   `experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42_final_30min/web/Code.gs`
   (identical to `src/evaluation/human/webapp_template/apps_script/Code.gs`).
4. Save (disk icon).
5. Choose **Deploy → New deployment → Select type: Web app**, then set
   * *Description*: `human eval collector`
   * *Execute as*: **Me**
   * *Who has access*: **Anyone**
6. Click **Deploy**, grant the permissions Google asks for (it warns because the script
   is unverified and yours; choose *Advanced → Go to … (unsafe)* → *Allow*).
7. Copy the **Web app URL**. It ends in `/exec`:
   `https://script.google.com/macros/s/AKfy…/exec`

Check it answers:

```bash
curl -L "https://script.google.com/macros/s/AKfy…/exec"
```

Expected: `{"ok":true,"service":"human-eval-collector"}`.

> Whenever you edit `Code.gs` later, use **Deploy → Manage deployments → edit → New
> version**, otherwise the old code stays live at the same URL.

## 3. Create the public repository for the site

GitHub Pages serves a static site for free. Create an **empty public** repository, for
example `dashboard-human-eval`, under your account. Do not initialise it with a README.

The published URL follows from the names, so you already know it now:

```text
https://<your-github-user>.github.io/dashboard-human-eval
```

Keep this repository separate from the thesis repository. It contains only the blinded
site, no predictions, no keys, no thesis text.

## 4. Export the site with the real URL and endpoint

```bash
python experiments/scripts/export_human_eval_web.py --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42_final_30min --base-url https://<your-github-user>.github.io/dashboard-human-eval --endpoint https://script.google.com/macros/s/AKfy…/exec --contact-email ramin.saeedivalashani@stud.hs-ruhrwest.de --institution "Ruhrwest University" --retention-months 12 --ethics-status "approved for use in this thesis"
```

The command prints the six personal rater links and writes them to
`…/seed_42_final_30min/web/rater_links.md`.

Re-running the export creates new tokens and therefore new links. To keep links stable
across re-exports, pass the salt from the existing key file:

```bash
python -c "import json;print(json.load(open('experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42_final_30min/web/unblinding_key.json'))['salt'])"
```

and add `--salt <value>` to the export command.

> **Back up the key.** `.gitignore` excludes everything under `experiments/results/`
> except `final/`, so the study directory and `web/unblinding_key.json` exist only on
> this machine and are not in git history. Without that file the collected ratings
> cannot be mapped back to methods. Copy it to a second location (for example an
> external drive or a private cloud folder) before sending the links out.

## 5. Publish the site

From the study's `web/site` directory:

```bash
cd experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42_final_30min/web/site
git init -b main
git add .
git commit -m "Publish blind rating site"
git remote add origin https://github.com/<your-github-user>/dashboard-human-eval.git
git push -u origin main
```

Then in the repository on github.com: **Settings → Pages → Source: Deploy from a
branch → Branch: `main` / `(root)` → Save**. The first build takes one to two minutes.

Publish the contents of `web/site` only. Never add `web/unblinding_key.json`,
`web/rater_links.md` or anything from `experiments/outputs/` to that repository.

## 6. Test it yourself before inviting anyone

1. Open the `rater_01` link from `rater_links.md`.
2. Rate one recommendation.
3. Open the spreadsheet: one row must appear within a few seconds, with
   `rater_id = rater_01`, a `unit_token`, and the six scores.
4. Delete that test row from the sheet afterwards (or leave it; the importer keeps the
   latest rating per rater and unit, and you can rate that unit again later).

If the row does not appear, see *Troubleshooting* below.

## 7. Send the links

Each rater gets exactly one link from `rater_links.md` — the link carries their ID, so
nobody has to choose one and two people cannot collide. Tell them:

* how long it takes (16 recommendations, planned total up to 30 minutes);
* that they can stop at any time and continue later with the same link, on any device;
* that the first screen explains the task and the six scales;
* that they should not discuss individual items with the other raters while rating.

The app itself repeats the instructions, so the message can stay short.

## 8. Watch progress

The spreadsheet is the live view: one row per rating, `rater_id` and `received_at` tell
you who is where. A quick count per rater:

```text
=QUERY(ratings!A:D, "select D, count(D) where D is not null group by D label count(D) 'ratings'", 1)
```

## 9. Import and analyse

Export the sheet (**File → Download → Comma-separated values**) and import it:

```bash
python experiments/scripts/import_human_eval_web.py --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42_final_30min --input ~/Downloads/ratings.csv
```

The importer unblinds the tokens with `web/unblinding_key.json`, rejects anything that
does not belong to the study, keeps the latest rating per rater and unit, and writes
`ratings/<rater>.jsonl` in exactly the schema the local Streamlit app produces. JSON
backup files that raters downloaded from the app can be passed in the same way
(`--input file1.json file2.json` or a directory). Use `--dry-run` to check first.

Then run the analysis:

```bash
python experiments/scripts/compute_irr.py --study-dir experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42_final_30min
```

It refuses to produce final numbers while ratings are missing; add `--allow-incomplete`
for an interim look and label such output as interim. The results land in
`analysis/`: rating completion, Krippendorff's ordinal alpha per dimension, per-method
means, per-item scores, the paired tests, and the derived chart-acceptability outcome.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| No rows appear in the sheet | The deployment is not public. In Apps Script: *Manage deployments → Who has access: Anyone*. Re-deploy as a **new version** and re-export with the `/exec` URL. |
| The app shows "Offline – ratings stored locally, retrying" | The endpoint is unreachable. Nothing is lost: ratings stay in the browser and are re-sent automatically; the rater can also download the backup file from the final screen and send it to you. |
| The final screen warns that ratings are unconfirmed | The browser could not read the collector's answer (a cross-origin restriction) although the data most likely arrived. Check the sheet; if the rows are there, ignore the warning. |
| A rater lost their browser data or switches device | Opening the same personal link restores their progress from the collector: already rated units are skipped. |
| A rater used the wrong link | Ratings are stored under the ID in the link. Either re-assign that ID to that person consistently, or ask them to redo the correct link; the importer rejects ratings for units that were never assigned to that ID. |
| Duplicate rows in the sheet | Expected after a retry. The importer keeps the latest rating per (rater, unit) and reports the number of duplicates. |
| `Unblinding key not found` on import | The export was re-run into a different directory, or the key was deleted. Re-export with the same `--salt`, or import with `--key <path>`. |
| `The rubric changed after this web bundle was exported` | `src/evaluation/human/rubric.py` was modified. Restore the rubric; ratings from two rubric versions are not comparable. |

## Running without any external service

If you prefer not to use a collector at all, omit `--endpoint`. The app then stores
everything in the browser and each rater downloads one JSON file on the final screen
and sends it to you. Import those files exactly as above. The trade-off: a rater who
never presses the download button hands in nothing, and progress cannot be recovered if
their browser data is cleared.

## Local rehearsal

The site is a plain directory, so it can be served locally before publishing:

```bash
python -m http.server 8765 --directory experiments/results/human_eval/dashboard_v4/qwen3_8_27b/seed_42_final_30min/web/site
```

Open `http://localhost:8765/?r=rater_01&t=<token>` with a token from `rater_links.md`.
The same configuration is available as the `human-eval-web` entry in `.claude/launch.json`.
