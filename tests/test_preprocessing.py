import pandas as pd

from chatlens import parse_chat, preprocess
from chatlens.preprocessing.extract import domain_of, extract_emojis, extract_urls
from chatlens.preprocessing.language import detect_language
from chatlens.preprocessing.normalize import normalize_text, tokenize


def test_no_rows_removed(chat_result):
    out = preprocess(chat_result.df)
    assert len(out) == len(chat_result.df)


def test_url_domain_extraction():
    assert extract_urls("see https://www.example.com/a?b=1, and www.foo.org.") == \
        ["https://www.example.com/a?b=1", "www.foo.org"]
    assert domain_of("https://www.Example.com/x") == "example.com"
    assert domain_of("www.foo.org") == "foo.org"


def test_emoji_extraction_including_zwj_and_skin_tone():
    got = extract_emojis("hi 😀👍🏽 👨‍👩‍👧 ok")
    assert got == ["😀", "👍🏽", "👨‍👩‍👧"]


def test_normalize_strips_urls_emoji_mentions():
    t = normalize_text("Hello WORLD https://x.com/a 😀 @923001234567 ok!")
    assert "http" not in t and "😀" not in t and "9230" not in t and t.split()[0] == "hello"


def test_urdu_diacritics_and_variants():
    assert normalize_text("كِتَاب") == "کتاب"


def test_hindi_matras_not_split():
    assert tokenize("नमस्ते दोस्त") == ["नमस्ते", "दोस्त"]


def test_tokenize_keeps_apostrophe_drops_digits():
    assert tokenize("don't stop 123 now_") == ["don't", "stop", "now"]


def test_language_detection_buckets():
    assert detect_language("how are you doing today") == "en"
    assert detect_language("kya haal hai yaar kal milte hain") == "hinglish"
    assert detect_language("आप कैसे हैं") == "hi"
    assert detect_language("آپ کیسے ہیں") == "ur"
    assert detect_language("hello آپ کیسے ہیں") == "mixed"
    assert detect_language("k") == "und"


def test_stopwords_are_language_specific():
    res = parse_chat("[01/01/2024, 10:00:00] A: the project is ready\n"
                     "[01/01/2024, 10:01:00] A: kya yaar project ready hai kal\n")
    d = preprocess(res.df)
    assert d.loc[0, "tokens_content"] == ["project", "ready"]
    assert d.loc[1, "language"] == "hinglish"
    assert d.loc[1, "tokens_content"] == ["project", "ready"]


def test_non_text_rows_have_no_features(chat_df):
    media = chat_df[chat_df["is_media"]]
    assert (media["n_words"] == 0).all() and (media["tokens"].str.len() == 0).all()
    assert not media["is_text"].any()


def test_counts_on_known_message():
    res = parse_chat("[01/01/2024, 10:00:00] A: hello! all good 😀 https://www.example.com/a\n")
    d = preprocess(res.df)
    assert d.loc[0, "n_words"] == 3 and d.loc[0, "n_emojis"] == 1
    assert d.loc[0, "domains"] == ["example.com"]
    assert isinstance(d, pd.DataFrame)
