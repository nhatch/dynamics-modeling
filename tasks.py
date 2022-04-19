import invoke


@invoke.task(optional=["apply"])
def lint(ctx, apply=False):
    """
    Lints code and provides suggestions.

    Args:
        apply (bool, optional): Apply changes in suggestions of isort and black.
            USE CAREFULLY, as it might result in deletion of important imports.
            Defaults to False.
    """
    DIR_NAME = "src/rosbag2torch"

    isort_flags = ["--diff", "--check-only"]
    black_flags = ["--diff"]

    if apply:
        isort_flags = black_flags = []

    isort_flags = " ".join(isort_flags)
    black_flags = " ".join(black_flags)

    ctx.run(
        f"isort {isort_flags} {DIR_NAME}", echo=True
    )
    result = ctx.run(
        f"black {black_flags} {DIR_NAME}", echo=True
    )

    ctx.run(f"flake8 {DIR_NAME}", echo=True)

    if "reformatted" not in result.stderr:
        ctx.run(f"mypy --no-incremental --cache-dir /dev/null {DIR_NAME}", echo=True)

