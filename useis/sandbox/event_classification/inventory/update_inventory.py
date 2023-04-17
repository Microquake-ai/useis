from uquake import read, read_inventory
from useis.core.project_manager import ProjectManager
from uquake import __package_name__ as ns

inventory = read_inventory('OT_old.xml')

for i, station in enumerate(inventory[0]):
    old_station_code = station.code
    station.historical_code = station.historical_code.replace('-', '_')
    if 'UG' in station.historical_code:
        station_number = int(station.historical_code.split('_')[0][4:])
        location_code = int(station.historical_code.split('_')[1][0])
        inventory[0][i].code = f'UG{station_number:02d}'
        if 'ACC' in station.historical_code:
            # Accelerometer
            channel_prefix = 'GN'
            if station_number in [1, 2, 3, 4, 5]:
                location_code = 1
        else:
            # Geophone
            channel_prefix = 'GP'
            if station_number in [1, 2, 3, 4, 5]:
                location_code = 2
    else:
        station_number = int(station.historical_code.split('_')[0][2:])
        location_code = int(station.historical_code.split('_')[1][0])
        inventory[0][i].code = f'SS{station_number:02d}'

    for extra_key in inventory[0][i].extra.keys():
        inventory[0][i].extra[extra_key].namespace = ns

    for j, channel in enumerate(inventory[0][i]):
        channel_code = channel.code
        if ('UG' in inventory[0][i].code) and (location_code == 3):
            location_code = 1
        inventory[0][i][j].location_code = f'0{location_code}'
        inventory[0][i][j].code = f'{channel_prefix}{channel_code.upper()}'
        for extra_key in inventory[0][i][j].extra.keys():
            inventory[0][i][j].extra[extra_key].namespace = ns

inventory.write('OT.xml', format='STATIONXML')

