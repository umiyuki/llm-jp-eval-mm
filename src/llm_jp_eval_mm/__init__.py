import os

from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

# 環境変数からPYTHONPATHを取得
pythonpath = os.getenv("PYTHONPATH")
if pythonpath:
    import sys

    sys.path.append(pythonpath)
