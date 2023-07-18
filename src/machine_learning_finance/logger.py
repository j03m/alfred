logging_level = 0


def error(*args):
    if logging_level >= 0:
        print(*args)


def info(*args):
    if logging_level >= 1:
        print(*args)


def verbose(*args):
    if logging_level >= 2:
        print(*args)


def debug(*args):
    if logging_level >= 3:
        print(*args)
