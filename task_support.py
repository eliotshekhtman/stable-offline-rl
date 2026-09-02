"""Single source of truth for supported task/source pairs."""


SUPPORTED_TASK_SOURCES = {
    "Reacher-v5": frozenset({"generated", "minari", "clean-minari"}),
    "HalfCheetah-v5": frozenset({"generated", "minari", "clean-minari"}),
    "Lift": frozenset({"robomimic"}),
    "Can": frozenset({"robomimic"}),
}

GENERATED_TASKS = frozenset(
    task for task, sources in SUPPORTED_TASK_SOURCES.items()
    if "generated" in sources
)
ROBOMIMIC_TASKS = frozenset(
    task for task, sources in SUPPORTED_TASK_SOURCES.items()
    if "robomimic" in sources
)


def require_supported_task(env_name: str, dataset_source: str) -> None:
    sources = SUPPORTED_TASK_SOURCES.get(env_name)
    if sources is None:
        supported = ", ".join(SUPPORTED_TASK_SOURCES)
        raise ValueError(
            f"Unsupported task {env_name!r}; supported tasks are: {supported}."
        )
    if dataset_source not in sources:
        available = ", ".join(sorted(sources))
        raise ValueError(
            f"Task {env_name!r} does not support dataset source "
            f"{dataset_source!r}; supported sources are: {available}."
        )
