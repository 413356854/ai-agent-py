"""报销审核 Agent：图导出、样例回归与 HTTP 接口。"""

from __future__ import annotations

import yaml
from fastapi.testclient import TestClient

import ai_agent_py.api.app as api_app
import ai_agent_py.reimburse.params as params_module
from ai_agent_py.reimburse import SAMPLES, SAMPLES_BY_ID, describe_graph, run_claim
from ai_agent_py.reimburse.rules import DATA_DIR


def test_graph_export_shape() -> None:
    graph = describe_graph()

    assert graph["node_count"] == 9
    assert graph["rule_count"] >= 10
    assert "graph TD" in graph["mermaid"]

    labels = {node["label"] for node in graph["nodes"]}
    assert {"文档接入", "判定融合", "人工复核", "归档"} <= labels

    edges = {(edge["source"], edge["target"]) for edge in graph["edges"]}
    # 三条并行检查分支
    assert {("extract", "hard_rules"), ("extract", "policy_check"), ("extract", "semantic_review")} <= edges
    # 汇合到判定融合
    assert {("hard_rules", "decide"), ("policy_check", "decide"), ("semantic_review", "decide")} <= edges
    # 中文标签替换后节点 id 仍然可定位
    assert "ingest(文档接入)" in graph["mermaid"]


def test_samples_hit_expected_status() -> None:
    for sample in SAMPLES:
        result = run_claim(sample["claim"])
        assert result["status"] == sample["expected"], sample["sample_id"]


def test_fan_in_barrier_runs_once() -> None:
    """三条并行检查汇合后，判定节点只执行一次。"""
    result = run_claim(SAMPLES[0]["claim"])
    nodes = [event["node"] for event in result["trace"]]
    assert nodes.count("decide") == 1
    assert nodes.count("hard_rules") == 1
    assert nodes.count("archive") == 1


def test_semantic_doubt_never_auto_rejects() -> None:
    """语义类 Finding（R4）只能转人工，不能直接退回。"""
    result = run_claim(SAMPLES[2]["claim"])  # C003：招待报成办公用品
    assert result["status"] == "转人工"
    assert any(finding["stage"] == "semantic" for finding in result["findings"])


def test_large_invoice_without_contract_warns() -> None:
    """R2.4 用 invoice_amount_requires_contract：缺合同只警告，不阻断。"""
    claim = SAMPLES_BY_ID["C007"]["claim"]
    result = run_claim(claim)
    assert result["status"] == "通过"
    assert "R2.4" in {finding["rule_id"] for finding in result["findings"]}

    with_contract = {**claim, "attachments": ["采购合同.pdf"]}
    result_with = run_claim(with_contract)
    assert "R2.4" not in {finding["rule_id"] for finding in result_with["findings"]}


def test_agents_api() -> None:
    client = TestClient(api_app.app)

    agents = client.get("/api/agents").json()
    assert agents[0]["id"] == "reimburse"
    assert agents[0]["node_count"] == 9

    graph = client.get("/api/agents/reimburse/graph").json()
    assert graph["node_count"] == 9
    assert graph["mermaid"] == graph["mermaid"]

    config = client.get("/api/agents/reimburse/config").json()
    assert config["param_status"] == "placeholder"
    assert len(config["rules"]) >= 10
    # 参数来源必须能指回具体文件，且规则要标出依赖的参数键
    assert config["param_file"].endswith("policy_params.yaml")
    assert {rule["id"] for rule in config["rules"] if rule["params"]} >= {"R3.1", "R3.2"}

    assert client.get("/api/agents/missing").status_code == 404

    samples = client.get("/api/agents/reimburse/samples").json()
    assert samples
    # 试跑页要展示单据原文，因此样例必须带 claim
    assert samples[0]["claim"]["invoices"]
    assert samples[0]["claim"]["total"]


def test_run_endpoint() -> None:
    client = TestClient(api_app.app)

    body = client.post("/api/agents/reimburse/run", json={"sample_id": "C002"}).json()
    assert body["status"] == "退回"
    assert {"R2.1"} <= {finding["rule_id"] for finding in body["findings"]}

    assert client.post("/api/agents/reimburse/run", json={}).status_code == 400
    assert client.post("/api/agents/reimburse/run", json={"sample_id": "NOPE"}).status_code == 404


def test_template_download_upload_roundtrip() -> None:
    """下载样板 → 原样上传 → 解析出的 claim 与样例一致 → 试跑结论一致。"""
    client = TestClient(api_app.app)

    blank = client.get("/api/agents/reimburse/template")
    assert blank.status_code == 200
    assert blank.content[:2] == b"PK"  # xlsx = zip

    sample = client.get("/api/agents/reimburse/samples/C002/download?fmt=xlsx")
    assert sample.status_code == 200

    upload = client.post(
        "/api/agents/reimburse/upload",
        files={
            "file": (
                "C002.xlsx",
                sample.content,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        },
    )
    assert upload.status_code == 200
    claim = upload.json()["claim"]
    assert claim["claim_id"] == "BX-2026-10-002"
    assert claim["total"] == "2300.00"
    assert claim["invoices"][0]["number"] == "24410003"
    assert claim["invoices"][0]["nights"] == 3
    assert claim["invoices"][0]["city_tier"] == "一线"

    result = client.post("/api/agents/reimburse/run", json={"claim": claim}).json()
    assert result["status"] == "退回"
    assert {"R2.1", "R3.1"} <= {finding["rule_id"] for finding in result["findings"]}


def test_json_sample_download() -> None:
    client = TestClient(api_app.app)
    body = client.get("/api/agents/reimburse/samples/C001/download?fmt=json")
    assert body.status_code == 200
    assert body.json()["claim_id"] == "BX-2026-10-001"


def test_upload_rejects_unsupported_file() -> None:
    client = TestClient(api_app.app)
    bad = client.post("/api/agents/reimburse/upload", files={"file": ("a.txt", b"hi", "text/plain")})
    assert bad.status_code == 400
    assert "xlsx" in bad.json()["detail"]


def test_update_params_persists_and_applies(tmp_path, monkeypatch) -> None:
    """参数改完要同时满足：落盘、刷新缓存、下一次试跑立即生效。"""
    target = tmp_path / "policy_params.yaml"
    target.write_text((DATA_DIR / "policy_params.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(params_module, "_store", params_module.PolicyParamsStore(target))

    client = TestClient(api_app.app)
    params = client.get("/api/agents/reimburse/config").json()["params"]
    # 参数文件是可改的（界面能改），测试不能假设某个固定值
    assert params["lodging_cap_per_night"]["一线"] > 0

    params["lodging_cap_per_night"]["一线"] = 9999
    updated = client.put("/api/agents/reimburse/config/params", json={"params": params})

    assert updated.status_code == 200
    assert updated.json()["params"]["lodging_cap_per_night"]["一线"] == 9999
    assert updated.json()["param_updated_at"]

    saved = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert saved["params"]["lodging_cap_per_night"]["一线"] == 9999

    # 800/晚 在新上限下不再超标，C002 只剩金额勾稽
    result = client.post("/api/agents/reimburse/run", json={"sample_id": "C002"}).json()
    assert "R3.1" not in {finding["rule_id"] for finding in result["findings"]}
    assert "R2.1" in {finding["rule_id"] for finding in result["findings"]}

    invalid = client.put(
        "/api/agents/reimburse/config/params",
        json={"params": {"lodging_cap_per_night": {"一线": 0}}},
    )
    assert invalid.status_code == 400
    assert "不能小于" in invalid.json()["detail"]

    # 校验失败不能破坏已落盘的文件
    assert yaml.safe_load(target.read_text(encoding="utf-8"))["params"]["lodging_cap_per_night"]["一线"] == 9999

    assert client.post("/api/agents/reimburse/config/reload").status_code == 200
