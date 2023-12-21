import matplotlib
import matplotlib.pyplot as plt

# Settings
plt.style.use('fivethirtyeight')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']                       

colors_hardcoded = [
  "#008fd5", 
  "#fc4f30", 
  "#e5ae38", 
  "#6d904f", 
  "#8b8b8b", 
  "#810f7c"
  ] # old color: #e5ae38, new color: #6d904f


init_color = colors[0]


color_mapping = matplotlib.colors.get_named_colors_mapping()