#! /usr/bin/python3

import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import os
import sys


def main():
    if len(sys.argv) < 2:
        print(f"Plot generator for HARP's CSV performance reports\n\nUsage: {sys.argv[0]} <INPUT.csv> [OUTPUT.png]")
        return -1

    input_file = sys.argv[1]
    if not os.path.exists(input_file) or not os.path.isfile(input_file):
        print(f"error: No such file or directory `{input_file}`")
        return -1

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_file)

    # Get unique variants
    variants = df.variant.unique().tolist()

    # Get benchmark name
    bench_name = df.kernel.unique()[0].upper()

    # Define some colors
    colors = [
        "#e78284", # red/Seq naive
        "#8caaee", # blue/Seq iter
        "#a6d189", # green/Rayon iter
        "#e5c890", # yellow/CL naive
        "#ca9ee6", # mauve/CL tiled
        "#f2d5cf", # rosewater/CUDA naive
        "#ef9f72", # peach/CUDA tiled
        "#414559", # gray/Rust-CUDA
    ]
    assert(len(colors) >= len(variants), "Not enough colors")

    # Define the figure and subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Average runtime",
            "Arithmetic intensity",
            "Memory bandwidth",
            "Computational performance",
        ],
    )

    # Plot the data for each subplot
    for i in range(4):
        row, col = i // 2 + 1, i % 2 + 1

        if i == 0:
            # Average runtime plot
            for v, variant in enumerate(variants):
                data = df[df.variant == variant]
                fig.add_trace(
                    go.Scatter(
                        x=data["elems_per_dim"],
                        y=data["avg_runtime"],
                        error_y=dict(type='data', array=data["stddev"], visible=True),
                        mode="lines",
                        name=variant,
                        legendgroup="group1",
                        line=dict(color=colors[v]),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(title_text="Size", row=row, col=col)
            fig.update_yaxes(title_text="Average runtime (ms)", row=row, col=col)
        elif i == 1:
            # Arithmetic intensity plot
            for v, variant in enumerate(variants):
                data = df[df["variant"] == variant]
                fig.add_trace(
                    go.Scatter(
                        x=data["FLOPs/Byte"],
                        y=data["GFLOP/s"],
                        mode="lines",
                        name=variant,
                        legendgroup="group1",
                        showlegend=False,
                        line=dict(color=colors[v]),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(title_text="FLOPs/Byte", type="log", row=row, col=col)
            fig.update_yaxes(title_text="GFLOP/s", type="log", row=row, col=col)
        elif i == 2:
            # Memory bandwidth plot
            for v, variant in enumerate(variants):
                data = df[df["variant"] == variant]
                fig.add_trace(
                    go.Scatter(
                        x=data["elems_per_dim"],
                        y=data["GiB/s"],
                        mode="lines",
                        name=variant,
                        legendgroup="group1",
                        showlegend=False,
                        line=dict(color=colors[v]),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(title_text="Size", row=row, col=col)
            fig.update_yaxes(title_text="GiB/s", row=row, col=col)
        elif i == 3:
            # Computational performance plot
            for v, variant in enumerate(variants):
                data = df[df["variant"] == variant]
                fig.add_trace(
                    go.Scatter(
                        x=data["elems_per_dim"],
                        y=data["GFLOP/s"],
                        mode="lines",
                        name=variant,
                        legendgroup="group1",
                        showlegend=False,
                        line=dict(color=colors[v]),
                    ),
                    row=row,
                    col=col,
                )
            fig.update_xaxes(title_text="Size", row=row, col=col)
            fig.update_yaxes(title_text="GFLOP/s", row=row, col=col)

    # Update layout
    fig.update_layout(
        title=f"Hardware-accelerated Rust {bench_name} performance metrics",
        title_font_size=16,
        legend=dict(x=1.1, y=0.5),
        legend_title="Implementation variants"
    )

    if len(sys.argv) == 3:
        output_file = sys.argv[2]
        fig.save(output_file)

    # Show the plot
    fig.show()

    return 0


if __name__ == "__main__":
    main()
