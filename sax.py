import numpy as np
import pandas as pd
import itertools
from saxpy.alphabet import cuts_for_asize
from saxpy.sax import ts_to_string
from bokeh.layouts import column, row
from bokeh.io import curdoc
from bokeh.models import DataTable, ColumnDataSource, Slider, BoxAnnotation, HoverTool, TableColumn, Div
from bokeh.plotting import figure
from bokeh.palettes import Category20


class RandomTimeSeries:
    '''
    Class to generate a random time-series with SAX-representation
    To initialize, provide number of observations and cardinality
    Observations are generated as a random Gaussian distribution with randomly generated mean and std deviation
    '''
    def __init__(self, n, cardinality = None):
        self.n = n
        self.values = np.random.randn(n) * np.random.randint(1, 20) + np.random.randint(-20, 20)
        self.mean = np.mean(self.values)
        self.std = np.std(self.values)
        self.norm_values = (self.values - self.mean) / self.std
        self.data = {'t': range(self.n), 'x': self.values}
        if cardinality:
            self.sax(cardinality)


    def sax(self, cardinality):
        '''
        Creates SAX representation of the time series
        :param cardinality: number of symbols to use in SAX representation
        '''
        self.cardinality = cardinality
        self.cuts = cuts_for_asize(self.cardinality)
        self.string = ts_to_string(self.norm_values, self.cuts)

        #denormalize cuts for correct vizualisaton
        self.cuts_den = self.cuts * self.std + self.mean
        self.data['symbol'] = list(self.string)
        self.sax_freq = self.generate_freq()


    def generate_freq(self):
        '''
        Generate frequency table for SAX symbols
        :return: pd.DataFrame
        '''
        freq = pd.Series(self.data['symbol']).value_counts()
        freq = freq.to_frame().reset_index()
        freq.columns = ['symbol', 'frequency']
        return freq


def get_color():
    '''
    Color generator function
    '''
    yield from itertools.cycle(Category20[20])


def gen_plot(ts):
    '''
    Generates plot for given time-series
    :param RandomTimeSeries ts:
    '''
    plot = figure(plot_width=1200, plot_height=400)
    plot.ygrid.visible = False
    plot.xgrid.visible = False

    data = ColumnDataSource(data=ts.data)
    plot.line('t', 'x', source=data)

    #adding color band for each symbol
    color = get_color()
    band = BoxAnnotation(top=ts.cuts_den[1], fill_color=next(color), fill_alpha=0.1)
    plot.add_layout(band)

    for i in range(2, ts.cardinality):
        band = BoxAnnotation(bottom=ts.cuts_den[i-1], top=ts.cuts_den[i], fill_color=next(color), fill_alpha=0.1)
        plot.add_layout(band)

    # Hover tool to check values for a data point
    plot.add_tools(HoverTool(
        tooltips=[
            ('value', '@x'),
            ('symbol', '@symbol'),
        ]))

    # updating SAX string and frequency table
    sax_string.text = ts.string
    freq_data.data = ts.sax_freq

    return plot


def set_cardinality(attr, old, new):
    '''
    Callback function to change cardinality for the time series by changing slider value
    '''
    ts.sax(new)
    plot = gen_plot(ts)

    global plot_row
    if plot_row.children:
        plot_row.children.pop()
    plot_row.children.append(plot)


def set_n(attr, old, new):
    '''
    Callback function to generate new time series of given length by changing slider value
    '''
    global ts
    ts = RandomTimeSeries(new, card_slider.value)
    plot = gen_plot(ts)
    global plot_row
    if plot_row.children:
        plot_row.children.pop()
    plot_row.children.append(plot)


### MAIN

# Control sliders
n_slider = Slider(start=2, end=10000, value=10, step=1, title="No. observations", width=900)
n_slider.on_change('value', set_n)

card_slider = Slider(start=2, end=20, value=3, step=1, title="Cardinality", width=250)
card_slider.on_change('value', set_cardinality)

# Default time-series
ts = RandomTimeSeries(n_slider.value, card_slider.value)

# SAX string representation
sax_label = Div(text='<b>SAX representation of Time Series:</b>')
sax_string = Div(style={'overflow-x':'scroll','width':'950px'})

# Frequency table
freq_data = ColumnDataSource()
freq_columns = [TableColumn(field='symbol', title='Symbol'),
                TableColumn(field='frequency', title='Frequency')]
freq_table = DataTable(source=freq_data, columns=freq_columns, width=200, index_position=None)

# Plot initialization
plot = gen_plot(ts)

# Setting layout
doc = curdoc()
curdoc().title = 'SAX Time Series'
plot_row = row(plot)

layout = column(
    row(n_slider, card_slider),
    plot_row,
    row(freq_table, column(
        sax_label,
        sax_string
    )
        )
)
doc.add_root(layout)




