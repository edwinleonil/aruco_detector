import pytest
from main import parse_camera_settings


def test_valid_values():
    gain, exposure = parse_camera_settings("2.0", "5000")
    assert gain == 2.0
    assert exposure == 5000.0


def test_integer_strings_accepted():
    gain, exposure = parse_camera_settings("3", "10000")
    assert gain == 3.0
    assert exposure == 10000.0


def test_non_numeric_gain_raises():
    with pytest.raises(ValueError, match="gain"):
        parse_camera_settings("abc", "5000")


def test_non_numeric_exposure_raises():
    with pytest.raises(ValueError, match="exposure"):
        parse_camera_settings("1.0", "xyz")


def test_negative_values_raise():
    with pytest.raises(ValueError):
        parse_camera_settings("-1", "5000")
    with pytest.raises(ValueError):
        parse_camera_settings("1.0", "-100")
