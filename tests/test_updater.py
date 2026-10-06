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
