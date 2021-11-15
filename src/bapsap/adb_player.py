# Pure Python adb client
from copy import deepcopy
from ppadb.client import Client as AdbClient

# For extracting color codes from screen shots
from .image.data_extraction import extract_ball_color_codes

# To implement Heuristic search in TubeState to find solution
from .tubestate import TubeState

import time


def initialize_client():
    adb_client = AdbClient(host="127.0.0.1", port=5037)
    print("ADB client version :", adb_client.version())
    return adb_client


def connect_to_device(client):
    print("Searching For connected Devices...")
    devices = client.devices()

    if len(devices) == 0:
        print("No devices found. Waiting for device to be connected...")
        while True:
            try:
                # check for any device available
                devices = client.devices()
                devices[0]
                break
            except (IndexError):
                # retry after one sec
                time.sleep(1)

    if len(devices) == 1:
        device = devices[0]
        print("Device found :", device.serial)
    elif len(devices) > 1:
        print("Multiple devices found. Please select one.")
        print(*(f"{i}. {dev.serial}" for i, dev in enumerate(devices)), sep="\n")
        device_id = int(input())
        device = devices[device_id]

    print("Connected to device:", device.serial)
    return device


def get_data_from_device(device):
    screen_shot = device.screencap()
    img_file_name = "level_image.png"
    with open(img_file_name, "wb") as fp:
        fp.write(screen_shot)

    return extract_ball_color_codes(img_file_name, 4, return_tube_coords=True)


def play_level(device):
    color_codes, tube_positions = get_data_from_device(device)
    tube_state = TubeState(color_codes)
    print(tube_state)
    # solution= tube_state.find_solution()
    solution = tube_state.find_solution_using_sahc()
    if not solution:
        print("No solution found")
        return

    print("Solution found.\nLength of solution: ", len(solution))

    solution_tube_positions = [
        (tube_positions[f], tube_positions[t]) for f, t in solution
    ]

    print("Playing...")
    for (fy, fx), (ty, tx) in solution_tube_positions:
        device.shell(f"input tap {fx} {fy}")
        device.shell(f"input tap {tx} {ty}")
    print("Level Complete!")


def play_multiple_levels(device, n_levels=5, start_wait=5, level_end_clicks=None):
    for _ in range(n_levels):
        time.sleep(start_wait)
        play_level(device)

        if level_end_clicks:
            for delay, (x, y) in level_end_clicks:
                time.sleep(delay)
                device.shell(f"input tap {x} {y}")


def main():
    adb_client = initialize_client()

    # connecting to a device from the client
    device = connect_to_device(adb_client)

    play_level(device)


if __name__ == "__main__":
    main()
