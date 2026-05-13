# Accessibility Guidelines for Dashboards

## 1. Color and Contrast

### Never Use Color as the Only Indicator

Do not rely on color alone to convey meaning. Users with color blindness cannot distinguish red from green.
Always combine color with a second indicator: shape, pattern, label, or icon.

**Bad:** A KPI card that turns red when below target — no other indicator.
**Good:** A KPI card that turns red AND shows a downward arrow icon AND the text "Below target".

### Use Accessible Color Palettes

Avoid red-green combinations — approximately 8% of men have red-green color blindness.

**Recommended accessible palettes:**

- Blue / Orange (safe for most color blindness types)
- Blue / Yellow
- Purple / Green
- Use tools like ColorBrewer (colorbrewer2.org) to pick accessible chart colors

### Ensure Sufficient Contrast Ratio

Text and chart elements must have a contrast ratio of at least 4.5:1 against the background (WCAG AA standard).
Use a contrast checker tool (e.g., WebAIM Contrast Checker) to verify.

**Minimum contrast ratios:**

- Normal text (< 18px): 4.5:1
- Large text (≥ 18px or bold ≥ 14px): 3:1
- Chart lines and bars: 3:1 against background

## 2. Text and Labels

### Use Clear, Readable Fonts

- Minimum font size for body text and labels: 12px
- Minimum font size for axis labels: 11px
- Avoid decorative or script fonts in dashboards
- Use sans-serif fonts (e.g., Inter, Roboto, Arial) for better screen readability

### Write Descriptive Chart Titles

Chart titles should describe what the chart shows, not just name the metric.
Screen readers read chart titles aloud — make them meaningful.

### Add Alt Text for Charts

If the dashboard is embedded in a web page or report, add alternative text (alt text) to every chart image.
Alt text should describe the key insight, not just the chart type.

**Example alt text:** "Bar chart showing monthly revenue by region. North region leads with $2.3M in Q4."

## 3. Interactive Elements

### Make Filters Keyboard-Accessible

All dropdown filters, date pickers, and buttons must be operable with a keyboard (Tab, Enter, Arrow keys).
Do not rely on mouse-only interactions.

### Provide Tooltips with Sufficient Contrast

Tooltip text must meet the same contrast requirements as regular text (4.5:1).
Tooltips should appear on both hover (mouse) and focus (keyboard).

### Avoid Auto-Refreshing Content

If the dashboard auto-refreshes, give users a way to pause or control the refresh rate.
Sudden content changes can be disorienting for users with cognitive disabilities or screen readers.

## 4. Layout and Cognitive Load

### Keep the Layout Predictable

Place navigation, filters, and KPI cards in the same position on every page of the dashboard.
Predictable layouts reduce cognitive load for all users, especially those with cognitive disabilities.

### Limit Information Density

Do not put too much information on one screen. A cluttered dashboard is hard to use for everyone,
but especially for users with attention or cognitive difficulties.

**Rule of thumb:** If a user needs more than 5 seconds to find the most important number, the dashboard is too complex.

### Use Clear Section Headings

Divide the dashboard into clearly labeled sections (e.g., "Revenue Overview", "Regional Breakdown", "Detailed Records").
Section headings help all users navigate quickly and help screen readers announce the structure.

## 5. Data Tables

### Make Tables Sortable and Filterable

Users should be able to sort any column and filter rows by keyword.
This is especially important for users who cannot easily scan visual charts.

### Use Proper Table Headers

Every column must have a clear header label.
Do not use merged cells or complex nested headers — they confuse screen readers.

### Provide a "Download as CSV" Option

Always offer a way to export table data as CSV.
This allows users with assistive technology to analyze the data in their preferred tool.
