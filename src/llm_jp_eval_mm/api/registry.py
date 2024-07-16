from llm_jp_eval_mm.api.model import lmms

MODEL_REGISTRY = {}


def register_model(*names):
    # either pass a list or a single alias.
    # function receives them as a tuple of strings

    def decorate(cls):
        print(f"Registering class: {cls.__name__} with names: {names}")
        print(f"cls class: {cls}")
        print(f"lmms class: {lmms}")

        for name in names:
            assert issubclass(cls, lmms), f"Model '{name}' ({cls.__name__}) must extend lmms class"

            assert (
                name not in MODEL_REGISTRY
            ), f"Model named '{name}' conflicts with existing model! Please register with a non-conflicting alias instead."

            MODEL_REGISTRY[name] = cls
        return cls

    return decorate


def get_model(model_name):
    try:
        return MODEL_REGISTRY[model_name]
    except KeyError:
        raise ValueError(
            f"Attempted to load model '{model_name}', but no model for this name found! Supported model names: {', '.join(MODEL_REGISTRY.keys())}"
        )


TASK_REGISTRY = {}  # Key: task name, Value: task ConfigurableTask class
GROUP_REGISTRY: dict[str, list[str]] = {}  # Key: group name, Value: list of task names
ALL_TASKS = set()  # Set of all task names and group names
func2task_index = {}  # Key: task ConfigurableTask class, Value: task name


def register_task(name):
    def decorate(fn):
        assert name not in TASK_REGISTRY, f"task named '{name}' conflicts with existing registered task!"

        TASK_REGISTRY[name] = fn
        ALL_TASKS.add(name)
        func2task_index[fn.__name__] = name
        return fn

    return decorate


def get_task(task_name, model_name):
    try:
        return TASK_REGISTRY[task_name](
            model_name=model_name
        )  # TODO choiszt the return result need to check " 'mmeConfigurableTask' object has no attribute '_instances'. Did you mean: 'instances'?"
    except KeyError:
        raise KeyError(f"Missing task {task_name}")
