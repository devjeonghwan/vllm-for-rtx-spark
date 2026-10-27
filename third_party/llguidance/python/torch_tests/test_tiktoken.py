import llguidance.tiktoken
import tiktoken


def test_tiktoken() -> None:
    enc = tiktoken.get_encoding("o200k_base")
    llt = llguidance.tiktoken.lltokenizer_from_encoding(enc)
    for s in [
            "Hello world!", "Hello world! こんにちは世界！", "wave 👋", "heart 👋💖",
            "1`a`b`c`d`e`f`g`h`i"
    ]:
        toks = llt.tokenize_str(s)
        print(llt.dbg_tokens(toks))
        assert llt.decode_str(toks) == s
    toks = llt.tokenize_bytes(b"\x8b")
    print(llt.dbg_tokens(toks))
    print(toks)
    assert len(toks) == 1
    assert llt.decode_bytes(toks) == b"\x8b"

    toks1 = llt.tokenize_str("<|endoftext|>")
    toks0 = llt.tokenize_str("<|endoftext|>", parse_special=False)
    assert toks1 == toks0
    assert len(toks0) > 1
    toks2 = llt.tokenize_str("<|endoftext|>", parse_special=True)
    assert len(toks2) == 1

    toks3 = llt.tokenize_str("a<|endoftext|>b", parse_special=True)
    print(llt.dbg_tokens(toks3))
    assert len(toks3) == 3
