from dotenv import load_dotenv
from os.path import join, dirname, abspath

dotenv_path = join(dirname(dirname(abspath(__file__))), ".env")
print("Loading .env file from", dotenv_path)
load_dotenv(dotenv_path)

from . import api, tasks
