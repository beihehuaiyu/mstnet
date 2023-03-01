"""
命令行参数解析器
"""

import argparse

def args_parser():
    """Common classifier application command-line arguments."""
    parser = argparse.ArgumentParser(
        description='EasyMIA command-line tools.')

    return parser