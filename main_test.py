from src.get_args import test_arguments
from src.test_model import run_test
#runs the entire testing process
if __name__ == "__main__":
    args = test_arguments()
    run_test(args)
