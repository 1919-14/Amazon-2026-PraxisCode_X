"""Compare ITRANS vs IAST vs OPTITRANS on real entity names."""

import sys
import unicodedata
import jellyfish
from indic_transliteration import sanscript

sys.stdout.reconfigure(encoding="utf-8")

UNICODE_RANGES = [
    (0x0900, 0x097F, "devanagari", sanscript.DEVANAGARI),
    (0x0980, 0x09FF, "bengali", sanscript.BENGALI),
    (0x0A00, 0x0A7F, "gurmukhi", sanscript.GURMUKHI),
    (0x0A80, 0x0AFF, "gujarati", sanscript.GUJARATI),
    (0x0B00, 0x0B7F, "oriya", sanscript.ORIYA),
    (0x0B80, 0x0BFF, "tamil", sanscript.TAMIL),
    (0x0C00, 0x0C7F, "telugu", sanscript.TELUGU),
    (0x0C80, 0x0CFF, "kannada", sanscript.KANNADA),
    (0x0D00, 0x0D7F, "malayalam", sanscript.MALAYALAM),
]

def detect_script(text: str):
    for ch in text:
        cp = ord(ch)
        for start, end, name, scheme in UNICODE_RANGES:
            if start <= cp <= end:
                return name, scheme
    return "latin", None

def romanize_with_scheme(text: str, scheme_name: str) -> str:
    s_name, s_scheme = detect_script(text)
    if not s_scheme:
        return text
    target = getattr(sanscript, scheme_name)
    try:
        rom = sanscript.transliterate(text, s_scheme, target)
        nfd = unicodedata.normalize("NFD", rom)
        clean = "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")
        clean = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in clean)
        return " ".join(clean.lower().split())
    except Exception as e:
        return text

test_samples = [
    ("राज इन्वेस्टमेंट्स प्राइवेट लिमिटेड", "Raj Investments Private Limited"),
    ("आनंद बिल्डर्स प्रा. लि.", "Anand Builders Pvt Ltd"),
    ("ಶ್ರೀ ಕೇರ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್", "Shree Care Private Limited"),
    ("तिरुपति मीडिया प्रा. लि.", "Tirupati Media Pvt Ltd"),
    ("अर्बन एग्रो एलएलपी", "Urban Agro LLP"),
    ("ઇન્ટરનેશનલ વિઝન ટેક્નોલોજીસ પ્રા. લિ.", "International Vision Technologies Pvt Ltd"),
    ("स्वस्तिक श्याम एंटरप्राइजेज प्राइवेट लिमिटेड", "Swastik Shyam Enterprises Private Limited"),
    ("લાઇફ એગ્રો પ્રાઇવેટ લિમિટેડ", "Life Agro Private Limited"),
    ("सन मार्केटिंग प्राइवेट लिमिटेड", "Sun Marketing Private Limited"),
    ("हाई लक्ष्मी इन्वेस्टमेंट", "High Laxmi Investment"),
    ("आनंद सर्विसेज प्राइवेट लिमिटेड", "Anand Services Private Limited"),
    ("शक्ति केयर प्राइवेट लिमिटेड", "Shakti Care Private Limited"),
    ("డైనమిక్ ప్రాపర్టీస్ ప్రైవేట్ లిమిటెడ్", "Dynamic Properties Pvt Ltd"),
    ("വൈറ്റ് കെയർ എൽഎൽപി", "White Care LLP"),
    ("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "Raj Investments LLP"),
]

for sc in ["ITRANS", "OPTITRANS", "IAST"]:
    print(f"\n=================== SCHEME: {sc} ===================")
    total_ov = 0
    total_tgt = 0
    for raw, target in test_samples:
        rom = romanize_with_scheme(raw, sc)
        m_rom = [jellyfish.metaphone(w) for w in rom.split() if len(w) >= 2]
        m_tgt = [jellyfish.metaphone(w) for w in target.lower().split() if len(w) >= 2]
        ov = len(set(m_rom) & set(m_tgt))
        total_ov += ov
        total_tgt += len(set(m_tgt))
        print(f"[{raw[:15]}] -> '{rom[:35]}' | meta: {m_rom[:3]} vs {m_tgt[:3]} (ov: {ov}/{len(set(m_tgt))})")
    print(f"TOTAL TOKEN METAPHONE OVERLAP: {total_ov} / {total_tgt} ({total_ov/total_tgt*100:.1f}%)")
