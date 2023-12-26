# Time Series Functions
# Package
import matplotlib.pyplot as plt

# Time Series
import statsmodels.api as sm

# Settings
plt.style.use('fivethirtyeight')

# VARIABLES
colors = [
  "#008fd5", 
  "#fc4f30", 
  "#e5ae38", 
  "#6d904f", 
  "#8b8b8b", 
  "#810f7c"
  ] # old color: #e5ae38, new color: #6d904f

def plot_ts_acf_pacf(data, x_label=None, y_label= None, title= None, color_plot = None, lags=40):
  '''
  Plot Time Series ACF and PCF
  '''
  # figure size
  ts_fig_size = (10, 4)
  acf_pacf_fig_size = (12,4)


  #Plot Time Series
  plt.figure(figsize= ts_fig_size, dpi=80)
  if color_plot is None:
    plt.plot( data, marker='o', markersize=2, linewidth=1)
  else:
    plt.plot( data, marker='o', markersize=2, linewidth=1, color=color_plot)
  plt.title(title)
  plt.xticks(rotation=45, ha='right')
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.show()


  # acf pacf plts
  fig, ax = plt.subplots(1,2,figsize= acf_pacf_fig_size)
  # acf
  sm.graphics.tsa.plot_acf(data.values.squeeze(), lags=lags, ax=ax[0], color=color_plot)
  #sm.graphics.tsa.plot_acf(data.values.squeeze(), ax=ax[0], color=color_plot)
  # pacf
  sm.graphics.tsa.plot_pacf(data, lags=lags, ax=ax[1], color=color_plot)
  #sm.graphics.tsa.plot_pacf(data, ax=ax[1], color=color_plot)

  plt.show()
  return None