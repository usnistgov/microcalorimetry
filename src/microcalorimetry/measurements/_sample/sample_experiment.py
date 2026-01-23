# -*- coding: utf-8 -*-
"""
This is a dummy experiment for use in debugging purposes.
"""

import time
import os


def main(output_dir='.', repeats=5, delay=1):
    print(os.path.abspath(output_dir))
    iterations = repeats

    try:
        for i in range(iterations):
            time.sleep(delay)
            print(' iteration number ', i, flush=True)

    except KeyboardInterrupt:
        print('Exiting clean')


if __name__ == '__main__':

    class args:
        def __init__(self):
            self.delay = 0.1
            self.repeats = 10
            self.output_dir = '.'

    main(args())
