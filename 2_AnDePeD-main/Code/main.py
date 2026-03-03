import read_files
import write_files
import data_handler
import offline_prep as prep
import online_use as use
import main_config as conf
import printer


if __name__ == '__main__':
    # create files in results to save optimal parameters achieved by offline preparation
    if conf.MODE == 'I':
        write_files.init_file_pandas(conf.OFFLINE_PREP_OPTIMAL_PARAMS_FILE.format('I'),
                                     ['algorithm', 'dataset', 'alpha_star', 'k_star', 'omega_star', 'l_vmd'])
    elif conf.MODE == 'II':
        write_files.init_file_pandas(conf.OFFLINE_PREP_OPTIMAL_PARAMS_FILE.format('II'),
                                     ['algorithm', 'dataset', 'modes_star_path'])

    algorithms = conf.ALGORITHMS

    dh = data_handler.DataHandler()

    for algorithm in algorithms:
        dh.reset()

        more_datasets_left = True
        while more_datasets_left:
            # OFFLINE PREPARATION
            datapath_offline, _ = dh.get_next_dataset('offline')

            alpha_star, k_star, l_vmd, modes_star_path, data_min, data_max =\
                prep.prepare_procedure(conf.MODE, algorithm, datapath_offline, conf.TEST_ID)

            data_parameters = [conf.L, alpha_star, k_star, l_vmd, modes_star_path, data_min, data_max]

            # ONLINE USE
            datapath_online, more_datasets_left = dh.get_next_dataset('online')
            dataname_online = datapath_online.split('/')[-1][:-4]
            printer.begin_online_use(algorithm, dataname_online)
            offline_dataset = read_files.\
                read_file_pandas(datapath_offline, column=conf.OFFLINE_COLUMN_NAME, to_numpy=True)
            online_dataset = read_files.\
                read_file_pandas(datapath_online, column=conf.ONLINE_COLUMN_NAME, to_numpy=True)
            onl_p = use.OnlineProcedure(conf.MODE, data_parameters, algorithm, dataname_online, offline_dataset)
            for value in online_dataset:
                onl_p.next_timestep(value)

            # save online results
            savepath = conf.RESULTS_DIR + f'/online_results_{conf.TEST_ID}_{algorithm}_{dataname_online}.csv'
            onl_p.export_saved_data(savepath)

            printer.end_online_use(algorithm, dataname_online)
