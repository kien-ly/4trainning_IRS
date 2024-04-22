import os
import csv
import numpy as np
import environment

import time
start_time = time.perf_counter()

num_antennas = 2               # "num_antennas", default=4, type=int, help='Number of antennas in the BS'
num_RIS_elements = 16         # "num_RIS_elements", default=4, type=int, help='Number of RIS elements'
num_users = 2                 # "num_users", default=4, type=int, help='Number of users'

power_t = 10                    # "power_t", default=0, type=float, help='Transmission power for the constrained optimization in dBm'

num_eps = 10000                   # "num_eps", default=10, type=int, help='Maximum number of episodes (default: 5000)
awgn_var = -169                # "awgn_var", default=-169, type=float, help='The noise power spectrum density in dBm/Hz (default: -169)
BW = 240000                    # "BW", default=240000, type=int, help='the transmission bandwidth in Hz (default: 240k)

env = environment.RIS_MISO(num_antennas,num_RIS_elements,num_users, AWGN_var=awgn_var)
file_name = f"{num_antennas}_{num_RIS_elements}_{num_users}_{power_t}_{num_eps}"

def generate_channel_data(num_channels):
    # Define the column names
    column_names = ['channel_number', 'G', 'H_r_all', 'H_d_all']
    # column_names = ['channel_number', 'G']

    # Create the channel data file
    with open(f'channel_csv/{file_name}.csv', 'w', newline='') as file:
        # Write the column names
        writer = csv.writer(file)
        writer.writerow(column_names)

        # Initialize the variable to hold the channel data
        channel_data = []

        # Generate the channel data for each channel
        for i in range(num_channels):
            # gen_channel = env._channel_generation()[0]
            gen_channel = env._channel_generation()

            # Convert the tuple to a list
            gen_channel_list = list(gen_channel)

            # Add the channel number to the channel data
            gen_channel_with_number = [str(i +1)] + gen_channel_list

            # Add the channel data to the channel data list
            channel_data.append(gen_channel_with_number)

        # Write the channel data to the file
        for row in channel_data:
            writer.writerow(row)

    print(f'{file_name}{num_channels} channel data files created.')

end_time = time.perf_counter()

if __name__ == "__main__":
    generate_channel_data(num_eps)
    print('time:  ',end_time - start_time)
