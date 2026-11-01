"""网络不通 github.com:443 时的推送预案：Git Data API 直推。

用法：.venv/Scripts/python.exe scripts/push_via_api.py
前置：gh auth 登录（取 token）；本地领先 origin/master 的提交需已 commit。

做法：blob → tree → commit（逐字段复刻本地提交，使 sha 一致）→ PATCH ref。
完成后本地与远端零分歧；网络恢复后可继续正常 git push。
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = "FrankLiWright/Fylgja"
API = f"https://api.github.com/repos/{REPO}"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True,
                                   encoding="utf-8", errors="replace").strip()


def main() -> int:
    token = subprocess.check_output(["gh", "auth", "token"], text=True).strip()
    hdr = {"Authorization": f"Bearer {token}",
           "Accept": "application/vnd.github+json",
           "X-GitHub-Api-Version": "2022-11-28"}

    def req(method: str, path: str, body=None):
        url = path if path.startswith("http") else API + path
        r = urllib.request.Request(
            url, method=method, headers=hdr,
            data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(r, timeout=60) as resp:
            return json.load(resp)

    local_head = _git("rev-parse", "HEAD")
    files = [f for f in _git("ls-files", "-z").split("\0") if f]  # -z：中文路径不转义
    if not files:
        print("无待推送文件")
        return 0

    # 本地提交元数据（尽力复刻）
    fmt = "%H%n%an%n%ae%n%aI%n%cn%n%ce%n%cI%n%B"
    meta = _git("log", "-1", f"--format={fmt}").splitlines()
    _, an, ae, aI, cn, ce, cI = meta[:7]
    message = "\n".join(meta[7:]).strip()

    entries = []
    for f in files:
        # 从 git 对象读取（规范化内容，树 sha 与本地完全一致）
        data = subprocess.check_output(["git", "cat-file", "blob", f"HEAD:{f}"])
        blob = req("POST", "/git/blobs", {
            "content": base64.b64encode(data).decode(),
            "encoding": "base64"})
        entries.append({"path": f, "mode": "100644", "type": "blob", "sha": blob["sha"]})

    parent = req("GET", "/git/ref/heads/master")["object"]["sha"]  # 远端真值
    tree = req("POST", "/git/trees", {"tree": entries})  # 完整树
    commit = req("POST", "/git/commits", {
        "message": message, "tree": tree["sha"], "parents": [parent],
        "author": {"name": an, "email": ae, "date": aI},
        "committer": {"name": cn, "email": ce, "date": cI},
    })
    sha = commit["sha"]
    req("PATCH", "/git/refs/heads/master", {"sha": sha})
    if sha == local_head:
        print(f"推送成功（sha 一致）：{sha}")
    else:
        print(f"推送成功（API sha={sha}，本地={local_head}——元数据有差异，git fetch 后核对）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
