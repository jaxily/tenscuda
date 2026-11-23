"""M4 env defaults without a card."""
import os

from tenscuda import env_flags


def test_compile_default_on():
    old = os.environ.pop("TENSCUDA_COMPILE", None)
    try:
        assert env_flags.compile_enabled() is True
        os.environ["TENSCUDA_COMPILE"] = "0"
        assert env_flags.compile_enabled() is False
    finally:
        if old is None:
            os.environ.pop("TENSCUDA_COMPILE", None)
        else:
            os.environ["TENSCUDA_COMPILE"] = old


def test_native_default_stays_off():
    assert env_flags.native_default_active() is False
    os.environ["TENSCUDA_NATIVE_DEFAULT"] = "1"
    try:
        assert env_flags.native_default_active() is True
    finally:
        del os.environ["TENSCUDA_NATIVE_DEFAULT"]