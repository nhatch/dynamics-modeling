from pathlib import Path
from typing import List
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
import argparse
import os

def filter_tf_warnings():
    """
    WEIRD STUFF TO SUPPRESS TENSORFLOW WARNINGS.
    Copied from: https://github.com/tensorflow/tensorboard/pull/3319
    """
    try:
        # Learn this one weird trick to make TF deprecation warnings go away.
        from tensorflow.python.util import deprecation
        return deprecation.silence()
    except (ImportError, AttributeError):
        def _null_context():
            """Pre-Python-3.7-compatible standin for contextlib.null_context."""
            yield
        _NULL_CONTEXT = _null_context()
        return _NULL_CONTEXT

def my_summary_iterator(path):
    with filter_tf_warnings():
        try:
            for r in tf_record.tf_record_iterator(path):
                yield event_pb2.Event.FromString(r)
        except:
            # This means that there was a Ctrl+C during the write.
            return [0]

def main():
    parser = argparse.ArgumentParser(
        "Small utility to remove interrupted tensorboard runs from a directory."
    )
    parser.add_argument('--logdir', type=str, default='runs/')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--min_epochs', type=int, default=2)

    args = parser.parse_args()

    to_remove: List[Path] = []
    for path in Path(args.logdir).rglob("events.out.tfevents*"):
        step = max([summary.step for summary in my_summary_iterator(str(path))])
        if step < args.min_epochs:
            to_remove.append(path)

    if args.dry_run:
        if len(to_remove) > 0:
            print("Would remove:\n\t- {}".format("\n\t- ".join(sorted(map(str, to_remove)))))
        else:
            print("Nothing to remove.")
        return

    # Remove files
    for path in to_remove:
        os.remove(str(path))

    # Remove empty directories
    for path in to_remove:
        if len(list(path.parent.rglob("*"))) == 0:
            path.parent.rmdir()

if __name__ == "__main__":
    main()
