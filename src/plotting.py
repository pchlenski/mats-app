"""All cribbed from https://colab.research.google.com/github/neelnanda-io/TransformerLens/blob/main/demos/Exploratory_Analysis_Demo.ipynb"""
# import plotly.express as px
import matplotlib.pyplot as plt
import torch
from transformer_lens import utils


def imshow(tensor, **kwargs):
    # px.imshow(
    return plt.imshow(utils.to_numpy(tensor), **kwargs)


def line(tensor, **kwargs):
    # px.line(
    return plt.line(y=utils.to_numpy(tensor), **kwargs)


def scatter(x, y, xaxis="", yaxis="", caxis="", **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    return plt.scatter(y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs)
