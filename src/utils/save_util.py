import os 
import datetime

def remove_file(fpath):
    if os.path.exists(fpath):
        os.remove(fpath)


def make_dir(dir):
    os.makedirs(dir, exist_ok=True)
    return dir


def make_save_dirs(args, prefix, suffix=None, with_time=False):
    time_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S") if with_time else ""
    suffix = suffix if suffix is not None else ""

    result_path = make_dir(os.path.join(args.result_path, prefix, args.exp_name, time_str))
    image_path = make_dir(os.path.join(result_path, "image"))
    log_path = make_dir(os.path.join(result_path, "log"))
    checkpoint_path = make_dir(os.path.join(result_path, "checkpoint"))
    sample_path = None
    sample_to_eval_path = None
    # if args.sample_to_eval:
    #     sample_to_eval_path = make_dir(os.path.join(result_path, "sample_to_eval", os.path.basename(args.resume_model)))
    print("create output path " + result_path)
    return image_path, checkpoint_path, log_path, sample_path, sample_to_eval_path