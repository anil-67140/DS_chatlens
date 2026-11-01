"""Compact built-in stopword lists. These are deliberately small and conservative:
they are NOT exhaustive linguistic resources. Hinglish/Roman-Urdu has no standard orthography,
so coverage is approximate. Lists can be extended via ``extra`` arguments in the pipeline."""

EN = frozenset("""a about above after again all am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further had has
have having he her here hers him his how i if in into is it its itself just me more most my no nor
not now of off on once only or other our out over own same she should so some such than that the
their them then there these they this those through to too under until up very was we were what
when where which while who whom why will with would you your yours im ive dont didnt cant wont
isnt u ur r n""".split())

CHAT_NOISE = frozenset("ok okay k lol haha hahaha hmm hm yeah yes yep nope oh ah".split())

# Romanised Hindi / Urdu (Hinglish / Roman Urdu) function words, approximate.
HINGLISH = frozenset("""hai hain hoon ho tha thi the hun hu nahi nahin na nhi ka ki ke ko se
me mein mai main mujhe mera meri mere tum tu tera teri tumhara aap aapka hum humara woh wo yeh ye
kya kyun kyu kaise kab kahan kaun kuch koi aur ya par pe lekin lakin toh to bhi hi bas abhi ab
phir fir kar kr karo karna kiya kia gaya gya gayi raha rahi rahe wala wali wale sab ek do ji haan
han acha accha theek thik yaar bhai kal aaj jab tab agar isliye kyunki tha""".split())

# Devanagari (Hindi)
HI = frozenset("""है हैं था थी थे हो हूँ का की के को से में पर और या भी तो नहीं ना यह ये वह वो
एक कि जो क्या कैसे कब कहाँ कौन कुछ कोई मैं मुझे मेरा मेरी हम तुम आप तू अब फिर बस लेकिन""".split())

# Arabic script (Urdu)
UR = frozenset("""ہے ہیں تھا تھی تھے ہو ہوں کا کی کے کو سے میں پر اور یا بھی تو نہیں نا یہ وہ
ایک کہ جو کیا کیسے کب کہاں کون کچھ کوئی میں مجھے میرا میری ہم تم آپ تو اب پھر بس لیکن
اس ان اپنا اپنی""".split())

BY_LANGUAGE = {
    "en": EN,
    "hinglish": EN | HINGLISH,
    "hi": HI | HINGLISH,
    "ur": UR,
}
ALL = EN | HINGLISH | HI | UR


def for_language(lang: str, *, include_chat_noise: bool = False,
                 extra: frozenset[str] = frozenset()) -> frozenset[str]:
    base = BY_LANGUAGE.get(lang, ALL)
    out = base | extra
    return out | CHAT_NOISE if include_chat_noise else out
