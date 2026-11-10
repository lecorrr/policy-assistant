"""Settings, configuration files, passwords and session tokens."""

import pytest
import yaml
from conftest import MODELS_FILE, USERS_FILE, make_settings
from pydantic import ValidationError

from policy_assistant import config
from policy_assistant.auth import SessionSigner, hash_password, verify_password
from policy_assistant.config import ConfigError, Price, load_models, load_users


def test_shipped_config_files_load(models, users):
    assert models.default == "local"
    assert models.memo == ["local", "gpt-6-luna", "claude-haiku-5-5"]
    assert models.models["local"].provider == "ollama"
    assert set(users.roles) == {"staff", "sales", "risk", "hr"}
    assert users.user("kateryna").roles == ("risk", "sales", "staff")


def test_every_user_gets_staff_and_unknown_logins_get_none(users):
    assert users.user("andrii").roles == ("staff",)
    assert users.user("nobody") is None


def test_settings_build_connection_strings_and_treat_empty_keys_as_unset():
    settings = make_settings(db_admin_password="a b'c", openai_api_key="", db_port=6000)
    assert "port=6000" in settings.admin_dsn and "user=policy_admin" in settings.admin_dsn
    assert "a b\\'c" in settings.admin_dsn  # quoted by libpq rules, so any password works
    assert settings.openai_api_key is None


def test_settings_reject_impossible_sizes():
    with pytest.raises(ValidationError, match="top_k"):
        make_settings(top_k=8, rerank_top=5)


def test_unknown_policy_variables_are_errors(monkeypatch):
    monkeypatch.setenv("POLICY_TOPK", "3")
    with pytest.raises(ConfigError, match="POLICY_TOPK"):
        config.load_settings()


def write_yaml(tmp_path, data, name="file.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def models_data():
    return yaml.safe_load(MODELS_FILE.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d["models"]["local"]["options"].update(think=False), "unknown options think"),
        (lambda d: d.update(default="missing"), "default: 'missing'"),
        (lambda d: d.update(memo=["local", "missing"]), "memo: 'missing'"),
        (lambda d: d.update(colour="red"), "colour"),
        (lambda d: d["models"]["local"]["price"].update(input=-1), "input"),
        (lambda d: d["models"]["claude-haiku-5-5"]["price"].pop("long_input"), "go together"),
        (lambda d: d["models"].update({"Bad Name": d["models"]["local"]}), "lower-case"),
    ],
)
def test_models_file_errors(tmp_path, change, message):
    data = models_data()
    change(data)
    with pytest.raises(ConfigError, match=message):
        load_models(write_yaml(tmp_path, data))


def test_prices_with_a_long_prompt_tier(models):
    haiku = models.models["claude-haiku-5-5"].price
    assert haiku.cost_usd(3000, 200) == pytest.approx((3000 * 0.10 + 200 * 0.50) / 1e6)
    assert haiku.cost_usd(120_000, 200) == pytest.approx((120_000 * 0.50 + 200 * 2.50) / 1e6)
    assert Price(input=0, output=0).cost_usd(10**6, 10**6) == 0


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda d: d["users"]["olena"].update(roles=["boss"]), "unknown roles boss"),
        (lambda d: d.update(roles=["sales", "risk", "hr"]), "must include 'staff'"),
        (lambda d: d["users"].update({"Olena!": d["users"]["olena"]}), "login"),
        (lambda d: d["users"]["olena"].update(password_hash="plain"), "password_hash"),
    ],
)
def test_users_file_errors(tmp_path, change, message):
    data = yaml.safe_load(USERS_FILE.read_text(encoding="utf-8"))
    change(data)
    with pytest.raises(ConfigError, match=message):
        load_users(write_yaml(tmp_path, data))


def test_missing_and_invalid_yaml_files(tmp_path):
    with pytest.raises(ConfigError, match="missing"):
        load_users(tmp_path / "none.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("users: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_users(bad)


def test_demo_passwords_match_the_readme(users):
    for login, entry in users.users.items():
        assert verify_password(f"demo-{login}", entry.password_hash), login


def test_password_hashes():
    stored = hash_password("correct horse")
    assert verify_password("correct horse", stored)
    assert not verify_password("wrong horse", stored)
    assert hash_password("correct horse") != stored  # salted
    for broken in ("", "scrypt$1$2", "bcrypt$x", stored.replace("scrypt", "md5", 1)):
        assert not verify_password("correct horse", broken)
    with pytest.raises(ValueError, match="8 characters"):
        hash_password("short")


def test_session_tokens():
    signer = SessionSigner("k" * 40, max_age_s=3600)
    token = signer.issue("olena", now=1000)
    assert signer.login_of(token, now=1000 + 3599) == "olena"
    assert signer.login_of(token, now=1000 + 3601) is None  # expired
    login, expires, signature = token.split(".")
    assert signer.login_of(f"iryna.{expires}.{signature}", now=1000) is None  # login swapped
    assert signer.login_of(f"{login}.{int(expires) + 99999}.{signature}", now=1000) is None
    assert SessionSigner("x" * 40, 3600).login_of(token, now=1000) is None  # other secret
    for junk in ("", "olena", "a.b.c", token + "x"):
        assert signer.login_of(junk, now=1000) is None
    with pytest.raises(ValueError, match="32 characters"):
        SessionSigner("short", 3600)
