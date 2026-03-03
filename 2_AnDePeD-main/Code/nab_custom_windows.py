import json


def create_custom_nab_windows_file(dataset: str, full_windows_file_path: str, export_windows_file_path: str):
    with open(full_windows_file_path) as full_wfile:
        full_windows = json.load(full_wfile)

    win_to_export = None
    for key in full_windows.keys():
        if dataset in key:
            win_to_export = full_windows[key]
    export_string = '{ \"' + dataset + '\": ' + str(win_to_export).replace('\'', '\"') + ' }'

    with open(export_windows_file_path, 'w') as export_wfile:
        export_wfile.write(export_string)

    return
