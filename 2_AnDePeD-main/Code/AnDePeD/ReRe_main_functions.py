from AnDePeD.ReRe import ReRe
import time as timelib
import gc


def run_rere(test_instance, progress_for_executions, executions_max, debug, export_filename): # , testrun_file):
    # create new instance of ReRe
    rere = ReRe.ReRe()
    # rere.TESTRUN_FILE = testrun_file
    rere.read_testrun_data()
    rere.init_timer()
    rere.init_offset_compensation()
    rere.init_auto_offset_compensation()
    rere.init_auto_ws_ap()

    # setting parameters automatically from the specified file
    if rere.OPERATION == 'file':
        rere.param_refresh(test_instance)
        rere.inst_num = test_instance
        rere.progress_for_executions = progress_for_executions
        rere.BATCH_STATUS_BAR = False
        rere.DEBUG = debug

    # load data file into the data frame and initialize operation parameters
    rere.load()

    if rere.OPERATION == 'file':
        rere.executions_max = executions_max
        executions_max[rere.inst_num] = rere.length
    rere.initialize_cons()

    # preprocess the dataset
    rere.preprocess()

    # algorithm starts now, starting timer
    timer_start = timelib.time()

    # initialize algorithm
    rere.initialize_rere()

    # dump parameters to results_yyyymmddhhmmss.csv
    if rere.TO_CSV:
        data_names, param_filename, result_filename = rere.init_to_csv(export_filename)
        rere.dump_hyperparameters(timer_start, param_filename)

    # MAIN ALGORITHM SECTION
    time = 0  # current timestep parameter
    while time < rere.length:
        if debug:
            print('RR(' + str(time) + '): Started timestep ' + str(time) + '.')
        # start time measurement
        rere.start_timestep()

        if debug:
            print('RR(' + str(time) + '): Current time saved.')

        # update the beginning of the sliding window
        rere.update_window_beginning(time)

        if debug:
            print('RR(' + str(time) + '): Window beginning updated.')

        # perform one timestep of the original ReRe algorithm
        rere.next_timestep(time)

        if debug:
            print('RR(' + str(time) + '): Next timestep done.')

        # perform offset compensation
        if rere.USE_OFFSET_COMP:
            rere.compensate_offset(time)
            if debug:
                print('RR(' + str(time) + '): Offset compensated.')

        # perform automatic tuning of offset compensation
        if rere.USE_AUTOMATIC_OFFSET:
            rere.auto_tune_offset(time)
            if debug:
                print('RR(' + str(time) + '): Offset compensation auto-tuned.')

        # perform automatic tuning of WINDOW_SIZE and AGE_POWER
        if rere.USE_AUTOMATIC_WS_AP:
            rere.auto_tune_ws_ap(time)
            if debug:
                print('RR(' + str(time) + '): WS and AP auto-tuned.')

        # stop time measurement and update averages
        rere.end_timestep(time)
        if debug:
            print('RR(' + str(time) + '): Time data saved and recalculated.')

        # dump results to a .csv file
        if rere.TO_CSV:
            rere.dump_results(time, data_names, result_filename)
            rere.write_time(timer_start, param_filename)
            if debug:
                print('RR(' + str(time) + '): Results exported.')

        # jump to the next timestep
        time += 1
        gc.collect()
