import json
import os

from notify import Feishu

commit_user = os.environ.get("GITHUB_ACTOR", "unknown")
commit_message = os.environ.get("COMMIT_MESSAGE", "unknown")
template = f'''\n 👶 {commit_user} \n 📝 {commit_message} \n 📦 {os.environ.get("RUNNER_NAME", "unknown")}_{os.environ.get("RUNNER_OS", 'unknown')}_{os.environ.get("RUNNER_ARCH", 'unknown')} \n 🚀 {os.environ.get("GITHUB_SERVER_URL", )}/{os.environ.get("GITHUB_REPOSITORY", )}/actions/runs/{os.environ.get("GITHUB_RUN_ID", )} \n'''

if __name__ == '__main__':
    fs = Feishu()
    if not os.path.exists("test_detail.json"):
        print("No Test Report Found!")
        fs.notify(template + "⚠ Test Ended but No Test Report Found!")
    else:
        report = json.load(open("test_detail.json"))
        test_nums = report.get("tests", 0) - report.get("disabled", 0)
        test_failures = report.get("failures", 0) + report.get("errors", 0)
        test_success = test_nums - test_failures
        if test_failures == 0:
            print(" ✅ Test Passed!")
            exit(0)
        else:
            message = template + f" ❌ Test Failed! \n 📊 {test_success}/{test_nums} \n"
            for ts in report.get("testsuites", []):
                for test in ts.get("testsuite", []):
                    if len(test.get("failures", [])) > 0:
                        failures = test.get("failures", [])
                        message += f"❌ {test.get('classname', 'unknown')}.{test.get('name', 'unknown')} \n"
