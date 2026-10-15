from videoredact import updater, __version__


def test_parse_version():
    assert updater.parse_version("v1.2.3") == (1, 2, 3)
    assert updater.parse_version("0.1") == (0, 1, 0)
    assert updater.parse_version("2.0.0-beta") == (2, 0, 0)


def test_is_newer():
    assert updater.is_newer("0.1.1", "0.1.0")
    assert updater.is_newer("0.2.0", "0.1.9")
    assert updater.is_newer("1.0.0", "0.99.99")
    assert not updater.is_newer("0.1.0", "0.1.0")
    assert not updater.is_newer("0.0.9", "0.1.0")
    assert not updater.is_newer(__version__)


def test_asset_pattern():
    assert updater.ASSET_RE.match("VideoRedact-0.1.1-Setup.exe").group(1) == "0.1.1"
    assert updater.ASSET_RE.match("videoredact-10.2.33-setup.EXE")
    assert not updater.ASSET_RE.match("VideoRedact-0.1.1-Portable.zip")
    assert not updater.ASSET_RE.match("Other-0.1.1-Setup.exe")


def test_parse_release_prefers_installer_asset():
    data = {"tag_name": "v9.9.9", "html_url": "https://github.com/x/y/releases/tag/v9.9.9", "body": "notes",
            "assets": [{"name": "SHA256SUMS.txt", "browser_download_url": "u1", "size": 1},
                       {"name": "VideoRedact-9.9.9-Setup.exe", "browser_download_url": "u2", "size": 123}]}
    info = updater.parse_release(data)
    assert info.version == "9.9.9" and info.url == "u2" and info.size == 123 and info.page.endswith("v9.9.9")
    bare = updater.parse_release({"tag_name": "v9.9.8"})
    assert bare.version == "9.9.8" and bare.url == "" and bare.page == updater.RELEASES_PAGE
    assert updater.parse_release({}) is None


def test_check_for_update_is_silent_offline(monkeypatch):
    def boom(*a, **k):
        raise OSError("no network")
    monkeypatch.setattr(updater.urllib.request, "urlopen", boom)
    assert updater.check_for_update(timeout=0.1) is None


def test_check_for_update_reports_newer_only(monkeypatch):
    monkeypatch.setattr(updater, "fetch_latest", lambda timeout=5.0: updater.UpdateInfo("99.0.0", "u", 1, "", "p"))
    assert updater.check_for_update().version == "99.0.0"
    monkeypatch.setattr(updater, "fetch_latest", lambda timeout=5.0: updater.UpdateInfo(__version__, "u", 1, "", "p"))
    assert updater.check_for_update() is None


def test_updater_never_installs():
    # Phase 5 rule: notify with a link only. Make sure nobody re-adds a download/launch path.
    assert not hasattr(updater, "download") and not hasattr(updater, "launch_installer")
    import subprocess
    assert "subprocess" not in open(updater.__file__, encoding="utf-8").read()


def test_version_copies_in_sync():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("bump_version", Path(__file__).parents[1] / "scripts" / "bump_version.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.current() == __version__
    assert mod.check() == []
    assert mod.notes(__version__)   # every released version has a changelog section
