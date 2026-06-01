# Chart Selection Guidelines

## 1. Line Chart — Use for Trends Over Time

Use a line chart when the x-axis represents a continuous time dimension (date, month, quarter, year).
Line charts make it easy to see how a metric rises, falls, or stays stable over time.

**Good for:** Revenue over months, daily active users, temperature over a year.
**Avoid when:** You have fewer than 3 data points or the x-axis is categorical (not time-based).

## 2. Bar Chart — Use for Category Comparisons

Use a bar chart when you want to compare a metric across discrete categories.
Horizontal bar charts work better when category labels are long.

**Good for:** Sales by region, headcount by department, revenue by product category.
**Avoid when:** You have more than 15–20 categories — consider grouping or filtering instead.

## 3. Stacked Bar Chart — Use for Composition Within Categories

Use a stacked bar chart when you want to show both the total and the breakdown by sub-category.

**Good for:** Revenue by region broken down by product type, budget allocation by department.
**Avoid when:** You have more than 5 sub-categories — the chart becomes hard to read.

## 4. Donut / Pie Chart — Use for Part-to-Whole Relationships

Use a donut or pie chart only when showing how parts make up a whole (shares, percentages).

**Good for:** Market share, budget allocation as percentages.
**Avoid when:** You have more than 5–6 slices. Many small slices make pie charts unreadable.
**Prefer donut over pie:** Donut charts are easier to read because the center can show a total value.

## 5. Scatter Plot — Use for Correlations and Relationships

Use a scatter plot when you want to show the relationship or correlation between two numeric variables.

**Good for:** Ad spend vs. conversions, price vs. demand, age vs. satisfaction score.
**Avoid when:** You have fewer than 20 data points — a table may be clearer.

## 6. Heatmap — Use for Patterns Across Two Dimensions

Use a heatmap when you want to show intensity or frequency across two categorical or time dimensions.

**Good for:** Sales by day of week and hour, support tickets by agent and category.
**Avoid when:** One of the dimensions has too many values (more than 20) — the cells become too small.

## 7. KPI Cards — Use for Headline Metrics

Use KPI cards (also called metric tiles or scorecards) to show the most important single numbers at a glance.
Always place KPI cards at the top of the dashboard.

**Good for:** Total revenue, conversion rate, number of open tickets, current headcount.
**Avoid when:** The metric changes so frequently that the card would be distracting.

## 8. Table — Use for Detailed Records

Use a table when users need to look up specific values, sort, or filter individual records.

**Good for:** Transaction lists, employee records, ticket details.
**Avoid when:** You want to show trends or comparisons — use a chart instead.

## General Rules

- Use the simplest chart type that communicates the message clearly.
- Never use 3D charts — they distort perception.
- Limit the number of chart types on one dashboard to 3–4 to maintain visual consistency.
- Always label axes with the metric name and unit (e.g., "Revenue (USD)").
