import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from AaltoAD.report_figures import downsample, style


def test_every_nth_step_series_keeps_multiples_of_step():
    data = pd.Series(range(20), index=range(20))
    kept = downsample.every_nth_step(data, step=5)
    assert list(kept.index) == [0, 5, 10, 15]
    assert list(kept.values) == [0, 5, 10, 15]


def test_every_nth_step_dataframe_keeps_multiples_of_step():
    data = pd.DataFrame({"a": range(10), "b": range(10, 20)}, index=range(10))
    kept = downsample.every_nth_step(data, step=3)
    assert list(kept.index) == [0, 3, 6, 9]


def test_every_nth_step_handles_negative_indices():
    data = pd.Series(range(6), index=[-30, -21, -20, -10, -5, 0])
    kept = downsample.every_nth_step(data, step=10)
    assert list(kept.index) == [-30, -20, -10, 0]


def test_every_nth_step_returns_empty_when_nothing_matches():
    data = pd.Series([1, 2, 3], index=[1, 3, 7])
    kept = downsample.every_nth_step(data, step=10)
    assert kept.empty


def test_plot_series_assigns_palette_colours_in_order_with_solid_lines():
    fig, ax = style.new_figure()
    df = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "c": [1, 1, 1]}, index=[0, 1, 2])
    style.plot_series(ax, df)
    lines = ax.get_lines()
    assert len(lines) == 3
    for line, color in zip(lines, style.PALETTE):
        assert line.get_color() == color
        assert line.get_linestyle() == "-"
    plt.close(fig)


def test_save_png_writes_readable_file_at_expected_size(tmp_path):
    fig, ax = style.new_figure()
    ax.plot([0, 1, 2], [0, 1, 0])
    out_path = tmp_path / "figure.png"
    style.save_png(fig, str(out_path))

    assert out_path.exists()
    image = mpimg.imread(str(out_path))
    width_px = image.shape[1]
    assert 3800 <= width_px <= 4600
