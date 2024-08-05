from llm_jp_eval_mm.configs import config


def log(text):
    if config.verbose:
        print(text)
