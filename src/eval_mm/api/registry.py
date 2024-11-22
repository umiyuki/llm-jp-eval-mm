TASK_REGISTRY = {}  # Key: task name, Value: task ConfigurableTask class
GROUP_REGISTRY: dict[str, list[str]] = {}  # Key: group name, Value: list of task names
ALL_TASKS = set()  # Set of all task names and group names
func2task_index = {}  # Key: task ConfigurableTask class, Value: task name


def register_task(name):
    def decorate(fn):
        assert (
            name not in TASK_REGISTRY
        ), f"task named '{name}' conflicts with existing registered task!"

        TASK_REGISTRY[name] = fn
        ALL_TASKS.add(name)
        func2task_index[fn.__name__] = name
        return fn

    return decorate


def get_task_cls(task_name):
    try:
        task_cls = TASK_REGISTRY[task_name]
        return task_cls
    except KeyError:
        raise KeyError(f"Missing task {task_name}")
