# Dashboard Design Guidelines

## 1. Layout and Hierarchy

### Put the Most Important KPIs at the Top

The top of the dashboard is where users look first. Place KPI cards (headline numbers) in the top row.
Below the KPI cards, show the main chart. Put supporting charts and detail tables at the bottom.

**Recommended layout pattern:**

```
[ KPI Card ]  [ KPI Card ]  [ KPI Card ]  [ KPI Card ]
[          Main Chart (full width or 2/3 width)       ]
[ Supporting Chart ]  [ Supporting Chart ]  [ Table   ]
```

### Use a Clear Visual Hierarchy

- Large numbers and bold text for KPI cards
- Medium-sized charts for the main analysis
- Smaller charts and tables for supporting detail
- Avoid placing too many elements at the same visual weight

### Limit the Number of Charts

A good dashboard has 4–8 visualizations. More than 10 charts on one screen overwhelms users.
If you need more, use tabs or drill-down pages.

## 2. Filters and Interactivity

### Always Provide a Date Range Filter

Most dashboards show time-based data. A date range picker at the top lets users focus on the period they care about.

### Add Dimension Filters for Key Categories

Provide dropdown filters for the most important dimensions:

- Region or country
- Product or product category
- Department or team
- Status (e.g., open/closed, active/inactive)

### Use Cross-Filtering

When a user clicks on a bar in a chart, the other charts on the dashboard should update to show only that selection.
This lets users explore data without navigating away.

### Provide Drill-Down

Allow users to click on a summary number to see the underlying records.
Example: Click on "Total Revenue: $1.2M" → see a table of individual transactions.

## 3. Labels and Scales

### Always Label Axes

Every chart axis must have a label that includes:

- The metric name (e.g., "Revenue")
- The unit (e.g., "USD", "%", "hours")

Example: "Revenue (USD)" not just "Revenue"

### Start Bar Chart Y-Axes at Zero

Never truncate the y-axis of a bar chart. Starting at a non-zero value exaggerates differences and misleads users.
Exception: Line charts may use a non-zero baseline when the range of variation is small.

### Use Consistent Scales

If you show the same metric in multiple charts, use the same scale on all of them.
Inconsistent scales make comparisons impossible.

### Format Large Numbers

- 1,000,000 → 1M
- 1,500 → 1.5K
- 0.1234 → 12.3% (for percentages)

## 4. Titles and Annotations

### Give Every Chart a Clear Title

The title should describe what the chart shows, not just the metric name.

- Bad: "Revenue"
- Good: "Monthly Revenue by Region (2023)"

### Add Annotations for Important Events

If there was a product launch, a policy change, or an anomaly, add a vertical line or annotation to the chart.
This helps users understand why a metric changed.

## 5. Consistency

### Use a Consistent Color Scheme

Pick 2–3 primary colors and use them consistently across all charts.
Do not use a different color for the same category in different charts.

### Use Consistent Fonts and Sizes

Use one font family throughout the dashboard.
Recommended sizes:

- KPI card numbers: 24–32px
- Chart titles: 14–16px
- Axis labels: 11–13px
- Tooltips: 11–12px

### Align Elements on a Grid

Use a grid layout so charts are aligned horizontally and vertically.
Misaligned elements look unprofessional and make the dashboard harder to scan.
