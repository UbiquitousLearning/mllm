import os

import requests


class Feishu:
    def __init__(self):
        self.env_token = os.environ.get('FEISHU_TOKEN', None)
        self.at_ = os.environ.get('FEISHU_AT', None)
        self.pre_str = "#MLLM_CI# "

    def notify(self, message: str):
        if self.env_token is None:
            return
        url = f'https://open.feishu.cn/open-apis/bot/v2/hook/{self.env_token}'
        if self.at_ is not None:
            self.pre_str += f'@{self.at_} \n'
        print(message)
        requests.post(url, json={"msg_type": "text", "content": {"text": self.pre_str + message}}, )
