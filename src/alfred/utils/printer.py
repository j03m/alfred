import sys
def print_in_place(message):
    # Print the message, return to start of line with \r, no newline
    sys.stdout.write(f"\r{message}")
    sys.stdout.flush()