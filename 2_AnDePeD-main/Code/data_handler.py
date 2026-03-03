import glob

import main_config as conf


class DataHandler:
    def __init__(self, offline_dir: str = conf.OFFLINE_DATA_DIR, online_dir: str = conf.ONLINE_DATA_DIR,
                 pp_offline_dir: str = conf.PP_OFFL_DATA_DIR, pp_online_dir: str = conf.PP_ONL_DATA_DIR):

        self.offline_dir = offline_dir
        self.online_dir = online_dir
        self.pp_offline_dir = pp_offline_dir
        self.pp_online_dir = pp_online_dir

        self.dataset_names = list()
        self.read_dataset_names_from_offline_dir()
        self.offline_dataset_index = 0
        self.online_dataset_index = 0

        return

    def read_dataset_names_from_offline_dir(self):
        data_names = glob.glob(self.offline_dir + '/*.csv')
        data_names = [n.replace('\\', '/') for n in data_names]
        data_names = [n.split('/')[-1][:-4] for n in data_names]

        self.dataset_names = data_names

        if len(self.dataset_names) == 0:
            print('ERROR: no dataset found in the directory ' + self.offline_dir)
        return

    def reset(self):
        self.dataset_names = list()
        self.read_dataset_names_from_offline_dir()

        self.offline_dataset_index = 0
        self.online_dataset_index = 0
        return

    def get_offline_dataset(self, index):
        name = self.dataset_names[index]
        candidates = glob.glob(self.offline_dir + '/' + name + '.csv')
        candidates = [c.replace('\\', '/') for c in candidates]
        if len(candidates) == 1:
            return candidates[0]
        else:
            print(f'ERROR: more than one offline dataset found with the same name: {name}')
            return 'ERR_not_unique'

    def get_online_dataset(self, index: int):
        name = self.dataset_names[index]
        path = self.online_dir + '/' + conf.ONLINE_FILE_STRUCTURES[2].format(name, '*')
        candidates = glob.glob(path)
        candidates = [c.replace('\\', '/') for c in candidates]

        if len(candidates) == 1:
            return candidates[0]

        elif len(candidates) == 0:
            print(f'ERROR: no online version found of the dataset {name}')
            return 'ERR_no_online_ver'

        else:
            chosen_candidate = candidates[0]
            chosen_id = get_date_id_from_path(chosen_candidate)
            print(f'INFO: more than one online candidate found for dataset {name}, '
                  f'choosing the one with id {chosen_id}')
            return chosen_candidate

    def get_next_dataset(self, mode: str = 'offline'):
        """
        :param mode: 'offline' or 'online'
        :return: dataset, dataset_name, more_left
        """
        if mode == 'offline':
            if self.offline_dataset_index >= len(self.dataset_names):
                print(f'ERROR: referencing higher dataset index than possible '
                      f'(OFFLINE index: {self.offline_dataset_index}, length: {len(self.dataset_names)})')
                return '', False
            else:
                path = self.get_offline_dataset(self.offline_dataset_index)
                more_left = self.offline_dataset_index < len(self.dataset_names) - 1
                self.offline_dataset_index += 1
                return path, more_left

        elif mode == 'online':
            if self.online_dataset_index >= len(self.dataset_names):
                print(f'ERROR: referencing higher dataset index than possible '
                      f'(ONLINE index: {self.online_dataset_index}, length: {len(self.dataset_names)})')
                return '', False
            else:
                path = self.get_online_dataset(self.online_dataset_index)
                more_left = self.online_dataset_index < len(self.dataset_names) - 1
                self.online_dataset_index += 1
                return path, more_left

        else:
            return '', False


def get_date_id_from_path(path):
    return path.split('-')[-1][:-4]
