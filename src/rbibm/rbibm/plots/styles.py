import matplotlib as mpl
import matplotlib.pyplot as plt
from tueplots import bundles

import os

_custom_styles = ["pyloric"]
_tueplot_styles = ["aistats2022", "icml2022", "jmlr2001", "neurips2021", "neurips2022"]
_mpl_styles = list(plt.style.available)

PATH = os.path.dirname(os.path.abspath(__file__))


def get_style(style, **kwargs):
    if style in _mpl_styles:
        return [style]
    elif style in _tueplot_styles:
        return [getattr(bundles, style)(**kwargs)]
    elif style in _custom_styles:
        return [PATH + os.sep + style + ".mplstyle"]
    elif style == "science":
        return ["science"]
    elif style == "science_grid":
        return ["science", {"axes.grid": True}]
    elif style is None:
        return None
    elif style == "icml_science_grid":
        return  [getattr(bundles, "icml2022")(**kwargs), "science", {"axes.grid": True}]
    else:
        return style


class use_style:
    def __init__(self, style, kwargs={}) -> None:
        super().__init__()
        self.style = get_style(style) +  [kwargs]
        self.previous_style = {}

    def __enter__(self):
        self.previous_style = mpl.rcParams.copy()
        if self.style is not None:
            plt.style.use(self.style)

    def __exit__(self, *args, **kwargs):
        mpl.rcParams.update(self.previous_style)
