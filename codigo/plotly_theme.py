from __future__ import annotations

import plotly.graph_objects as go


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="black",
        plot_bgcolor="black",
        font=dict(color="white", size=14),
        legend=dict(
            bgcolor="rgba(0,0,0,0.35)",
            bordercolor="rgba(255,255,255,0.15)",
            borderwidth=1,
        ),
        margin=dict(l=60, r=40, t=60, b=50),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig

