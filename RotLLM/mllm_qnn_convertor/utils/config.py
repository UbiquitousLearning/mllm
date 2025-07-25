class ConfigDict(dict):
    """配置字典，支持点号访问"""
    def __getattr__(self, key):
        # try:
        #     return self[key]
        # except KeyError:
        #     # 让getattr的默认值机制生效
        #     raise AttributeError(key)
        return self.get(key, None)
        
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __dir__(self):
        return list(self.keys()) + list(super().__dir__())
    