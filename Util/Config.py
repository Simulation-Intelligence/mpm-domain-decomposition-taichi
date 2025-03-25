import json

# ------------------ 配置模块 ------------------
class Config:
    def __init__(self, path=None, data=None):
        if path is not None:
            with open(path) as f:
                self.data = json.load(f)
        elif data is not None:
            self.data = data
        else:
            self.data = {}
    
    def get(self, key, default=None):
        return self.data.get(key, default)
