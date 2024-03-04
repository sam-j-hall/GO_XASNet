import torch
import numpy as np
from bokeh.plotting import figure
from bokeh.palettes import HighContrast3

def pred_spec(model, index, test_dataset, graphnet):
    # --- Set the model to evaluation mode
    device = 'cpu'

    x, edge_index = test_dataset[index[0]].x, test_dataset[index[0]].edge_index
    batch = torch.repeat_interleave(torch.tensor(0), x.shape[0])

    model.to(device)
    model.eval()
    with torch.no_grad():
        if graphnet == True:
            pred = model(index[1])
        else:
            pred = model(x, edge_index, batch)

    # --- Access the predicted output for the single graph
    true_spectrum = test_dataset[index[0]].spectrum.cpu().numpy()
    predicted_spectrum = pred.cpu().numpy()
    predicted_spectrum = predicted_spectrum.reshape(-1)
    
    return predicted_spectrum, true_spectrum

def calculate_rse(prediction, true_result):
    
    del_E = 20 / len(prediction)

    numerator = np.sum(del_E * np.power((true_result - prediction),2))

    denominator = np.sum(del_E * true_result)

    return np.sqrt(numerator) / denominator

def bokeh_spectra(ml_spectra, true_spectra):
    p = figure(
    x_axis_label = 'Photon Energy (eV)', y_axis_label = 'arb. units',
    x_range = (280,300),
    width = 350, height = 350,
    outline_line_color = 'black', outline_line_width = 2
    )

    p.toolbar.logo = None
    p.toolbar_location = None
    p.min_border = 25

    # x-axis settings
    p.xaxis.ticker.desired_num_ticks = 3
    p.xaxis.axis_label_text_font_size = '24px'
    p.xaxis.major_label_text_font_size = '24px'
    p.xaxis.major_tick_in = 0
    p.xaxis.major_tick_out = 10
    p.xaxis.minor_tick_out = 6
    p.xaxis.major_tick_line_width = 2
    p.xaxis.minor_tick_line_width = 2
    p.xaxis.major_tick_line_color = 'black'
    p.xaxis.minor_tick_line_color = 'black'
    # y-axis settings
    p.yaxis.axis_label_text_font_size = '24px'
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.yaxis.major_label_text_color = None
    # grid settings
    p.grid.grid_line_color = 'grey'
    p.grid.grid_line_alpha = 0.3
    p.grid.grid_line_width = 1.5
    p.grid.grid_line_dash = "dashed"

    # plot data
    x = np.linspace(280,300,200)
    p.line(x, true_spectra, line_width=3, line_color=HighContrast3[0], legend_label='True')
    p.line(x, ml_spectra, line_width=3, line_color=HighContrast3[1], legend_label='ML Model')

    # legend settings
    p.legend.location = 'bottom_right'
    p.legend.label_text_font_size = '20px'

    return p