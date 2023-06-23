import pytest

from perception_dataset.utils.gen_tokens import generate_token


def test_generate_token():
    assert type(generate_token(16, "bytes")) == bytes
    assert len(generate_token(16, "bytes")) == 16
    assert len(generate_token(32, "bytes")) == 32
    assert generate_token(16, "bytes") != generate_token(16, "bytes")

    assert type(generate_token(16, "hex")) == str
    assert len(generate_token(16, "hex")) == 32
    assert len(generate_token(32, "hex")) == 64
    assert generate_token(16, "hex") != generate_token(16, "hex")

    assert type(generate_token(16, "urlsafe")) == str
    assert len(generate_token(16, "urlsafe")) == 22
    assert len(generate_token(32, "urlsafe")) == 43
    assert generate_token(16, "urlsafe") != generate_token(16, "urlsafe")

    with pytest.raises(ValueError) as e:
        generate_token(16, "noexist")
    e.match("Invalid argument 'mode'='noexist'")

    with pytest.raises(ValueError) as e:
        generate_token(15, "hex")
    e.match("nbytes 15 is too short. Give >= 16.")
