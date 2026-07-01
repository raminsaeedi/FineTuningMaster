# External Real-Brief Evaluation Set — Provenance

Provenance and reproducibility record for [`data/eval/real_briefs/items.jsonl`](../../data/eval/real_briefs/items.jsonl):
an **external, non-circular** test set of dashboard *briefs* reverse-engineered from real, publicly
documented BI dashboards. This is the **first tranche (10 items)**; it will extend to 20–40 after review.

## Scientific framing (four questions)
- **What is tested?** How the four methods perform on **real, unseen** dashboard problems (external validity).
- **With which data?** 10 hand-authored `DashboardBrief` items derived from public BI examples; **input-only**.
- **With which metric?** **L2** schema/format/completeness/validity, **L3** dashboard realism, and **L4**
  human quality/usefulness ratings (the validity anchor). **L1 Top-1 chart accuracy is NOT claimed** for
  this set (briefs carry no `task_type`/gold solution).
- **How to interpret?** Non-circular external evidence for "better/more useful" claims; small n → report
  conservatively with CIs.

## Non-circularity & separation
- Briefs capture the **problem only** (audience, goals, KPIs, columns, constraints). The original
  dashboards' **chart types, encodings, layout, visual hierarchy, and colors are deliberately excluded**.
- Disjoint by construction from all training/augmentation sources (synthetic `gold.jsonl`,
  ChartGPT/nvBench/Quda). No label or label-generation lineage is shared with training.

## Leakage control
- `items.jsonl` `extra` is limited to `{"provenance_id": ...}`. `item_id`s are **neutral** (`rb_001`…),
  carrying no vendor/source name. All source identity (vendor, URL, snapshot) lives **only in this file**,
  so passing a full brief object to a prompt cannot reveal the source.

## Authoring protocol
For each source: `users` ← documented audience/persona; `goals` ← stated purpose/business questions;
`kpis` ← documented measures/KPIs; `columns` ← documented data model (`name` snake_case;
`dtype ∈ {datetime, categorical, numeric}`); `constraints` ← documented refresh/limits, else `null`.
Original brief text is authored from facts; source datasets/screenshots are **not** redistributed.

## Legend
- **confidence**: `high` = brief fields taken directly from the documented source page/schema;
  `medium` = generalized from a public demo (specific dashboard snapshot still to be pinned).
- **license/usage**: `recorded` = terms known; `cite-and-ask` = confirm before any data reuse.
- **implicit_task_coverage**: AUDIT-ONLY. **Not used as model input and not used as scoring gold.**
- All items accessed **2026-07-01**. Wayback snapshots to be pinned during the extension pass; the
  Power BI Learn pages are versioned/stable (page `ms.date` 2025-09-11).

## Provenance table

| item_id | domain | source_family | source_name | source_url | source_license/usage | confidence | implicit_task_coverage (audit-only) | derivation_notes | excluded (visual solution) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rb_001 | Retail | Power BI sample | Retail Analysis Sample | https://learn.microsoft.com/en-us/power-bi/create-reports/sample-retail-analysis | MS learning sample (obviEnce, anonymized); facts only (`recorded`) | high | trend, comparison, deviation, ranking, part_to_whole | Audience/KPIs (this-vs-last-year sales, variance %, sales/sq ft, gross margin, new stores) and fields (chain, district, manager, category, store, fiscal month) from the doc | pie/bubble/line/area charts, tiles, layout |
| rb_002 | Supply chain | Power BI sample | Supplier Quality Analysis Sample | https://learn.microsoft.com/en-us/power-bi/create-reports/sample-supplier-quality | MS learning sample (obviEnce); facts only (`recorded`) | high | ranking, comparison, part_to_whole, trend, distribution | Two stated metrics (defect qty, downtime min) + objectives (best/worst suppliers, plants); fields material type, vendor, plant, defect type/category | combo/treemap/map charts, layout |
| rb_003 | HR | Power BI sample | Human Resources Sample | https://learn.microsoft.com/en-us/power-bi/create-reports/sample-human-resources | MS learning sample (obviEnce); facts only (`recorded`) | high | trend, comparison, part_to_whole, composition, deviation | Objectives (who you hire, biases, voluntary separations) + measures (new hires vs SPLY, actives, separations, bad hires) + dims region/ethnicity/age/gender/reason | waterfall/pie/donut/line charts, layout |
| rb_004 | IT finance | Power BI sample | IT Spend Analysis Sample | https://learn.microsoft.com/en-us/power-bi/create-reports/sample-it-spend | MS learning sample (obviEnce); facts only (`recorded`) | high | deviation, comparison, trend, composition, ranking | Plan-vs-actual variance focus, Var Plan %, Var LE3 %, dims sales region, IT area/sub area, cost element group, business area; annual plan + quarterly LE | bar/column charts, slicers, layout |
| rb_005 | Finance | Power BI sample | Customer Profitability Sample | https://learn.microsoft.com/en-us/power-bi/create-reports/sample-customer-profitability | MS learning sample (obviEnce); facts only (`recorded`) | high | comparison, deviation, ranking, part_to_whole, trend, correlation | CFO audience; GM%, revenue vs budget variance, revenue by manager/segment, YoY growth; dims executive, product, customer, region, industry, scenario | bubble/treemap/KPI/line charts, mobile page, layout |
| rb_006 | Procurement | Power BI sample | Procurement Analysis Sample | https://learn.microsoft.com/en-us/power-bi/create-reports/sample-procurement | MS learning sample (obviEnce); facts only (`recorded`) | high | trend, part_to_whole, ranking, comparison, correlation | Stated questions (top vendors, biggest categories, discounts & timing); total invoice, discount %; dims category/sub-category, country, city, tier, vendor | line/map/treemap/combo charts, layout |
| rb_007 | Marketing | Power BI sample | Sales and Marketing Sample (VanArsdel) | https://learn.microsoft.com/en-us/power-bi/create-reports/sample-sales-and-marketing | MS learning sample (obviEnce); facts only (`recorded`) | high | trend, comparison, part_to_whole, deviation | CMO audience; market share %, R12M share, total units, YTD variance %, sentiment score/gap; dims manufacturer, region, segment, category, month | line/column/treemap charts, tiles, layout |
| rb_008 | Finance | Power BI sample | Financial Sample workbook | https://learn.microsoft.com/en-us/power-bi/create-reports/sample-financial-download | MS learning sample; facts only (`recorded`) | high | part_to_whole, comparison, trend, ranking | Documented table: segment, country, product, discount band, units sold, sale price, gross sales, discounts, sales, COGS, profit, date | none authored (no visuals documented) |
| rb_009 | Retail | Tableau | Sample - Superstore (bundled dataset) | Bundled with Tableau Desktop; widely mirrored (e.g. public.tableau.com) — exact stable URL to confirm | Tableau sample data; redistribution terms unclear (`cite-and-ask`) | high | trend, ranking, correlation, part_to_whole, comparison | Canonical Superstore schema: order date, ship mode, segment, region, state, category, sub-category, sales, quantity, discount, profit; audience = regional sales managers | any chart types/dashboards built on it |
| rb_010 | IT operations | Grafana Play | Grafana Play public observability demo | https://play.grafana.org | Grafana AGPLv3; public demo, data terms unspecified (`cite-and-ask`) | medium | trend, distribution, deviation, comparison, (weak) flow | Generalized SRE golden signals (request rate, error %, latency p50/p95/p99, CPU saturation) over time/service/endpoint; **specific dashboard snapshot to pin on extension** | panel/gauge/heatmap charts, layout |

## Coverage summary (this tranche)
- **Source families:** Power BI samples (8), Tableau Superstore (1), Grafana Play (1).
- **Domains:** retail, supply chain, HR, IT finance, finance/profitability, procurement, marketing,
  financial sales, IT operations.
- **Implicit `task_type` coverage (audit-only):** `trend`, `comparison`, `composition`, `distribution`,
  `correlation`, `ranking`, `deviation`, `part_to_whole` are all represented. **`flow` is only weakly
  represented** (request path in rb_010) — target a sales-pipeline / marketing-funnel brief in the extension.

## Open items before extending to 20–40
- Pin **Wayback Machine** snapshots for all source URLs (especially Grafana Play and the Tableau page).
- Confirm the exact stable URL and redistribution terms for the Tableau Superstore dataset (`cite-and-ask`).
- Pin a specific Grafana Play dashboard and verify its panels (rb_010 is currently `medium` confidence).
- Add `flow`-oriented briefs and broaden domains (e.g. healthcare, energy, web/product analytics).
- Optionally add Looker Studio template-gallery items (3rd family) on extension.
