# Chart Features (Toggles)

All features are off by default to preserve the current look and behavior. Use the checkboxes in the chart options row to enable them when needed.

- Zoom: Mouse wheel to zoom on the x-axis, drag to pan. Double-click the chart or click "Reset Zoom" to reset.
- Log: Switch the main y-axis to logarithmic scale. Automatically falls back to linear for non-positive values.
- SMA 50 / SMA 200: Simple moving average overlays computed on the client from the index series.
- Drawdown: Adds a drawdown (%) line on a secondary y-axis.
- Events: Draws vertical markers from /api/events. Edit events/events.json to update.
- Metrics: Shows period performance statistics under the legend.
- Benchmarks: Compare against SPY, QQQ, ACWI, EEM, XLV, IBB. Normalized to the index start value for the selected period.
- Export CSV: Downloads a CSV of the visible series for the selected period.
- Permalink: URL query parameters reflect the current configuration for sharing.
