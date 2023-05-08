import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def generate_sensor_positions_plot(df, all_regions, bg_image_dir):

    tallinn_map = plt.imread(bg_image_dir)

    image_width = tallinn_map.shape[1] / 100 * 2
    image_height = tallinn_map.shape[0] / 100 * 2

    lon_min = df['longitude'].min()
    lon_max = df['longitude'].max()
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()

    colors = ['red', 'blue', 'green', 'black']

    fig, ax = plt.subplots(figsize=(image_width, image_height))

    # Create basemap with specified boundaries
    m = Basemap(projection='merc', resolution='h',
                llcrnrlon=lat_min, llcrnrlat=lon_min,
                urcrnrlon=lat_max, urcrnrlat=lon_max)
    m.imshow(tallinn_map, interpolation='lanczos', origin='upper')

    # Plot sensor locations
    for i, r in enumerate(all_regions):
        x, y = m(df[df['region'] == r]['latitude'].values, df[df['region'] == r]['longitude'].values)
        m.scatter(x, y, c=colors[i], marker='o', s=10)

    ax.set_xlabel('latitude')
    ax.set_ylabel('longitude')

    legend = ["Region " + str(r) for r in all_regions]
    ax.legend(legend)
    plt.show()