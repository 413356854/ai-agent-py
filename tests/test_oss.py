"""私域 OSS：签名、服务层校验与 /api/oss 接口（全部离线，用假 OSS 客户端）。"""

from __future__ import annotations

import base64
import hashlib
import hmac
from urllib.parse import parse_qs, quote, urlsplit

import pytest
from fastapi.testclient import TestClient

import ai_agent_py.api.app as api_app
from ai_agent_py.alioss.client import OssClient, OssListedObject, OssListResult
from ai_agent_py.alioss.service import OssImageService
from ai_agent_py.api import oss as oss_api
from ai_agent_py.config import Settings

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
ACCESS = "test-access-token"
OSS_HOST = "https://agent3.oss-cn-shenzhen.aliyuncs.com"


class FakeOssClient:
    """内存版 OSS：覆盖上传 / 列举 / 删除 / 预签名（不联网）。"""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

    async def put_object(self, key: str, data: bytes, content_type: str) -> str:
        self.objects[key] = (data, content_type)
        return "etag-1"

    async def delete_object(self, key: str) -> None:
        if key not in self.objects:
            from ai_agent_py.errors import OssObjectNotFound

            raise OssObjectNotFound(f"OSS 对象不存在: {key}")
        del self.objects[key]

    async def list_objects(
        self, *, prefix: str, max_keys: int = 100, marker: str | None = None
    ) -> OssListResult:
        keys = sorted(key for key in self.objects if key.startswith(prefix))
        if marker:
            keys = [key for key in keys if key > marker]
        page = keys[:max_keys]
        return OssListResult(
            objects=[
                OssListedObject(
                    key=key,
                    size=len(self.objects[key][0]),
                    last_modified="2026-11-05T05:25:43.000Z",
                    etag="etag-1",
                )
                for key in page
            ],
            truncated=len(keys) > max_keys,
            next_marker=page[-1] if len(keys) > max_keys and page else None,
        )

    def presign_url(
        self,
        key: str,
        *,
        expires_in: int | None = None,
        response_content_disposition: str | None = None,
    ) -> tuple[str, int]:
        ttl = expires_in or 3600
        query = "OSSAccessKeyId=AK&Expires=1&Signature=sig"
        if response_content_disposition:
            query += f"&response-content-disposition={quote(response_content_disposition, safe='')}"
        return f"{OSS_HOST}/{key}?{query}", ttl


@pytest.fixture
def oss_settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path / "raw",
        oss_bucket="agent3",
        oss_endpoint="https://oss-cn-shenzhen.aliyuncs.com",
        oss_region="cn-shenzhen",
        oss_access_key_id="AK",
        oss_access_key_secret="SK",
        oss_prefix="image3",
        oss_access_token=ACCESS,
        oss_url_expires=3600,
        oss_max_file_mb=1,
        oss_allowed_suffixes=".png,.jpg,.jpeg",
    )


@pytest.fixture
def fake_client() -> FakeOssClient:
    return FakeOssClient()


@pytest.fixture
def service(oss_settings: Settings, fake_client: FakeOssClient) -> OssImageService:
    return OssImageService(oss_settings, client=fake_client)  # type: ignore[arg-type]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, service: OssImageService) -> TestClient:
    monkeypatch.setattr(oss_api, "get_service", lambda request: service)
    monkeypatch.setattr(oss_api, "get_settings", lambda: service.settings)
    return TestClient(api_app.app)


def _upload(client: TestClient, name: str = "cat.png", data: bytes = PNG):
    return client.post(
        "/api/oss/images",
        headers={"X-OSS-Access": ACCESS},
        files={"file": (name, data, "image/png")},
    )


# ---- 签名 / 配置 ---------------------------------------------------------
@pytest.mark.asyncio
async def test_authorization_matches_oss_v1_signature(oss_settings: Settings) -> None:
    async with OssClient(oss_settings) as oss:
        date = "Thu, 05 Nov 2026 05:40:50 GMT"
        authorization = oss.authorization("PUT", "image3/a.png", "image/png", date)
    string_to_sign = f"PUT\n\nimage/png\n{date}\n/agent3/image3/a.png"
    expected = base64.b64encode(
        hmac.new(b"SK", string_to_sign.encode(), hashlib.sha1).digest()
    ).decode()
    assert authorization == f"OSS AK:{expected}"


@pytest.mark.asyncio
async def test_presign_url_signs_content_disposition(
    oss_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("ai_agent_py.alioss.client.time.time", lambda: 1000)
    async with OssClient(oss_settings) as oss:
        url, ttl = oss.presign_url(
            "image3/a.png",
            expires_in=3600,
            response_content_disposition='attachment; filename="a.png"',
        )
    query = parse_qs(urlsplit(url).query)
    assert ttl == 3600
    assert query["Expires"] == ["4600"]
    assert query["response-content-disposition"] == ['attachment; filename="a.png"']
    string_to_sign = (
        "GET\n\n\n4600\n"
        '/agent3/image3/a.png?response-content-disposition=attachment; filename="a.png"'
    )
    expected = base64.b64encode(
        hmac.new(b"SK", string_to_sign.encode(), hashlib.sha1).digest()
    ).decode()
    assert query["Signature"] == [expected]


def test_settings_normalize_oss_values(tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path / "raw",
        oss_bucket=" Agent3 ",
        oss_endpoint="agent3.oss-cn-shenzhen.aliyuncs.com/",
        oss_prefix="/image3/",
        oss_access_key_id="AK",
        oss_access_key_secret="SK",
        oss_access_token="token",
    )
    assert settings.oss_bucket == "agent3"
    assert settings.oss_endpoint == "https://agent3.oss-cn-shenzhen.aliyuncs.com"
    assert settings.oss_host == "agent3.oss-cn-shenzhen.aliyuncs.com"
    assert settings.oss_prefix == "image3"
    assert settings.oss_suffixes[0] == ".jpg"
    assert settings.describe()["oss_access_token"] == "set"


# ---- 服务层 ---------------------------------------------------------------
@pytest.mark.asyncio
async def test_service_upload_list_delete(service: OssImageService) -> None:
    uploaded = await service.upload("测试.png", PNG)
    assert uploaded.key.startswith("image3/")
    assert uploaded.key.endswith(".png")
    assert uploaded.size == len(PNG)
    assert uploaded.url.startswith(f"{OSS_HOST}/image3/")
    assert uploaded.download_url.startswith(f"{OSS_HOST}/image3/")
    assert "response-content-disposition=" in uploaded.download_url
    assert uploaded.expires_in == 3600

    listed = await service.list_objects(limit=10)
    assert listed.count == 1
    assert listed.objects[0].key == uploaded.key
    assert listed.objects[0].url.startswith(f"{OSS_HOST}/image3/")

    assert await service.delete(uploaded.key) == uploaded.key
    listed = await service.list_objects(limit=10)
    assert listed.count == 0


@pytest.mark.asyncio
async def test_service_rejects_bad_files(service: OssImageService) -> None:
    with pytest.raises(Exception, match="只支持上传图片"):
        await service.upload("note.txt", b"hello")
    with pytest.raises(Exception, match="不是有效的"):
        await service.upload("fake.png", b"not an image")
    with pytest.raises(Exception, match="图片过大"):
        await service.upload("big.png", PNG + b"\x00" * (1024 * 1024))


@pytest.mark.asyncio
async def test_service_rejects_other_prefix(service: OssImageService) -> None:
    with pytest.raises(Exception, match="只允许访问"):
        service.presign("other/secret.png")
    with pytest.raises(Exception, match="非法路径"):
        service.presign("image3/../other/secret.png")


# ---- /api/oss -------------------------------------------------------------
def test_overview(client: TestClient) -> None:
    response = client.get("/api/oss/overview")
    body = response.json()
    assert body["configured"] is True
    assert body["access_configured"] is True
    assert body["bucket"] == "agent3"
    assert body["prefix"] == "image3"
    assert body["access_required"] is True
    assert ".png" in body["accepted_suffixes"]
    assert response.headers["cache-control"] == "no-store"


def test_overview_sets_httponly_session_cookie(client: TestClient) -> None:
    response = client.get("/api/oss/overview")
    set_cookie = response.headers.get("set-cookie", "")
    lowered = set_cookie.lower()
    assert "oss_session=" in lowered
    assert "httponly" in lowered
    assert "samesite=strict" in lowered
    assert "path=/api/oss" in lowered

    # 浏览器 Cookie 自动携带：上传 / 删除都不需要再手填口令。
    upload = client.post("/api/oss/images", files={"file": ("cat.png", PNG, "image/png")})
    assert upload.status_code == 201
    key = upload.json()["key"]
    assert client.delete(f"/api/oss/images/{key}").status_code == 204


def test_tampered_session_cookie_is_rejected(client: TestClient) -> None:
    client.get("/api/oss/overview")
    client.cookies.clear()
    client.cookies.set("oss_session", "tampered", path="/api/oss")
    response = client.post("/api/oss/images", files={"file": ("cat.png", PNG, "image/png")})
    assert response.status_code == 401


def test_upload_requires_access(client: TestClient) -> None:
    response = client.post("/api/oss/images", files={"file": ("cat.png", PNG, "image/png")})
    assert response.status_code == 401
    wrong = client.post(
        "/api/oss/images",
        headers={"X-OSS-Access": "wrong"},
        files={"file": ("cat.png", PNG, "image/png")},
    )
    assert wrong.status_code == 401


def test_upload_returns_oss_direct_urls(client: TestClient) -> None:
    response = _upload(client)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["key"].startswith("image3/")
    assert body["url"].startswith(f"{OSS_HOST}/image3/")
    assert body["download_url"].startswith(f"{OSS_HOST}/image3/")
    assert "response-content-disposition=" in body["download_url"]
    assert body["expires_in"] == 3600
    assert "signed_url" not in body
    assert "/api/oss/images/" not in body["url"]


def test_presign_endpoint_returns_new_direct_urls(client: TestClient) -> None:
    body = _upload(client).json()
    response = client.get(
        "/api/oss/presign",
        params={"key": body["key"]},
        headers={"X-OSS-Access": ACCESS},
    )
    assert response.status_code == 200, response.text
    presigned = response.json()
    assert presigned["key"] == body["key"]
    assert presigned["url"].startswith(f"{OSS_HOST}/image3/")
    assert presigned["download_url"].startswith(f"{OSS_HOST}/image3/")
    assert presigned["expires_in"] == 3600

    no_access = client.get("/api/oss/presign", params={"key": body["key"]})
    assert no_access.status_code == 401
    other_prefix = client.get(
        "/api/oss/presign",
        params={"key": "other/x.png"},
        headers={"X-OSS-Access": ACCESS},
    )
    assert other_prefix.status_code == 400


def test_list_and_delete(client: TestClient) -> None:
    first = _upload(client, "a.png").json()
    second = _upload(client, "b.png").json()

    listing = client.get("/api/oss/objects", headers={"X-OSS-Access": ACCESS})
    assert listing.status_code == 200
    body = listing.json()
    assert body["count"] == 2
    assert {item["key"] for item in body["objects"]} == {first["key"], second["key"]}
    assert all(item["url"].startswith(OSS_HOST) for item in body["objects"])
    assert all("response-content-disposition=" in item["download_url"] for item in body["objects"])

    deleted = client.delete(f"/api/oss/images/{first['key']}", headers={"X-OSS-Access": ACCESS})
    assert deleted.status_code == 204

    listing = client.get("/api/oss/objects", headers={"X-OSS-Access": ACCESS}).json()
    assert listing["count"] == 1


def test_upload_rejects_non_image(client: TestClient) -> None:
    response = client.post(
        "/api/oss/images",
        headers={"X-OSS-Access": ACCESS},
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_delete_requires_access(client: TestClient) -> None:
    body = _upload(client).json()
    assert client.delete(f"/api/oss/images/{body['key']}").status_code == 401


def test_unconfigured_returns_503(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        data_dir=tmp_path / "raw",
        oss_bucket=None,
        oss_access_token=None,
    )
    service = OssImageService(settings, client=FakeOssClient())  # type: ignore[arg-type]
    monkeypatch.setattr(oss_api, "get_service", lambda request: service)
    monkeypatch.setattr(oss_api, "get_settings", lambda: settings)
    client = TestClient(api_app.app)

    overview = client.get("/api/oss/overview")
    assert overview.json()["configured"] is False
    assert "set-cookie" not in overview.headers
    response = client.post(
        "/api/oss/images",
        headers={"X-OSS-Access": "anything"},
        files={"file": ("cat.png", PNG, "image/png")},
    )
    assert response.status_code == 503
