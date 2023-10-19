import json
import os

import requests


class Feishu:
    def __init__(self):
        self.env_token = os.environ.get('FEISHU_TOKEN', None)
        self.at_ = os.environ.get('FEISHU_AT', None)
        self.pre_str = "#MLLM_CI# "

    def notify(self, message: str):
        print(message)
        if self.env_token is None:
            return
        url = f'https://open.feishu.cn/open-apis/bot/v2/hook/{self.env_token}'
        if self.at_ is not None:
            try:
                at_s = json.loads(self.at_)
                commit_user = os.environ.get("GITHUB_ACTOR", "")
                if commit_user != "":
                    print("Actor: ", commit_user)
                    user_id = at_s[commit_user]
                    self.pre_str += f' <at user_id="{user_id}">{commit_user}</at> '
            except Exception as e:
                print(e)
        text = requests.post(url, json={"msg_type": "text", "content": {"text": self.pre_str + message}}, ).text
        # print(text)


class PRComment:
    def __init__(self):
        self.token = os.environ.get('GITHUB_TOKEN', None)
        self.enable = "request" in os.environ.get('GITHUB_EVENT_NAME', "")
        self.pr_number = os.environ.get('PR_NUMBER', None)
        self.repo = os.environ.get('GITHUB_REPOSITORY', None)
        self.endpoint = os.environ.get('GITHUB_API_URL', None)

    def notify(self, message: str):
        if self.token is None or self.enable is False or self.pr_number is None:
            return
        url = f'{self.endpoint}/repos/{self.repo}/issues/{self.pr_number}/comments'
        print(requests.post(url, json={"body": message}, headers={"Authorization": f"token {self.token}"}, ).text)
