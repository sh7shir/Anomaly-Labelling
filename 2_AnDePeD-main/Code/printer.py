DIVIDER_LENGTH = 100
DIVIDER_CHAR = '#'


def template_beg():
    print()
    print(DIVIDER_CHAR * DIVIDER_LENGTH)


def template_mid(text):
    print(DIVIDER_CHAR + ' ' + text)


def template_end():
    print(DIVIDER_CHAR * DIVIDER_LENGTH)
    print()


def begin_offline_preparation(algorithm: str, dataset: str, mode: str):
    template_beg()
    template_mid('BEGINNING OFFLINE PREPARATION FOR ' + algorithm + ' ON ' + dataset)
    template_mid('MODE: ' + mode)
    template_end()


def end_offline_preparation(algorithm: str, dataset: str, mode: str, best_params: str):
    template_beg()
    template_mid('FINISHED OFFLINE PREPARATION FOR ' + algorithm + ' ON ' + dataset)
    template_mid('MODE: ' + mode)
    template_mid('BEST PARAMETERS: ' + str(best_params))
    template_end()


def step_offline_preparation(current: int, out_of: int):
    template_beg()
    template_mid('FINISHED OPTIMISATION STEP {}/{}'.format(current, out_of))
    template_end()


def begin_online_use(algorithm, dataset):
    template_beg()
    template_mid('BEGINNING ONLINE USE OF ' + algorithm + ' ON ' + dataset)
    template_end()


def end_online_use(algorithm, dataset):
    template_beg()
    template_mid('FINISHED ONLINE USE OF ' + algorithm + ' ON ' + dataset)
    template_end()
