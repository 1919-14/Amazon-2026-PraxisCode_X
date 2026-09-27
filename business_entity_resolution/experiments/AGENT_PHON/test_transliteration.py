"""Test transliteration schemes and speed on real India names."""

import sys
import time
import jellyfish
from indic_transliteration import sanscript

sys.stdout.reconfigure(encoding="utf-8")

samples = [
    (sanscript.DEVANAGARI, "राज इन्वेस्टमेंट्स प्राइवेट लिमिटेड", "Raj Investments Private Limited"),
    (sanscript.TAMIL, "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "Raj Investments LLP"),
    (sanscript.KANNADA, "ರಾಜ್ ಇನ್ವೆಸ್ಟ್ಮೆಂಟ್ಸ್", "Raj Investments"),
    (sanscript.TELUGU, "రాజ్ ఇన్వెస్ట్మెంట్స్", "Raj Investments"),
    (sanscript.GUJARATI, "શક્તિ અર્બન પ્રોડક્ટ્સ પ્રાઇવેટ લિમિટેડ", "Shakti Urban Products Private Limited"),
    (sanscript.BENGALI, "গোল্ড প্রডিউসার স্টোর্স লিমিটেড", "Gold Producer Stores Limited"),
    (sanscript.MALAYALAM, "ശക്തി ഇംപെക്സ് പ്രൈവറ്റ് ലിമിറ്റഡ്", "Shakti Impex Private Limited"),
    (sanscript.ORIYA, "ଭିଜନ୍ ଟେକ୍ନୋଲୋଜିସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍", "Vision Technologies Private Limited"),
]

for scheme_name in ["ITRANS", "IAST", "HK", "OPTITRANS"]:
    scheme = getattr(sanscript, scheme_name)
    print(f"\n=== SCHEME: {scheme_name} ===")
    for src_scheme, txt, latin_target in samples:
        try:
            rom = sanscript.transliterate(txt, src_scheme, scheme)
            rom_clean = "".join(c for c in rom if c.isalnum() or c.isspace()).lower()
            target_clean = "".join(c for c in latin_target if c.isalnum() or c.isspace()).lower()
            rom_meta = [jellyfish.metaphone(w) for w in rom_clean.split() if len(w) >= 2]
            target_meta = [jellyfish.metaphone(w) for w in target_clean.split() if len(w) >= 2]
            print(f"[{src_scheme}] {txt} -> {rom_clean}")
            print(f"   metaphone: {rom_meta} vs target: {target_meta}")
        except Exception as e:
            print(f"[{src_scheme}] Error: {e}")

# Measure speed of transliterating 10,000 strings
t0 = time.time()
n_iter = 5000
for _ in range(n_iter):
    sanscript.transliterate("राज इन्वेस्टमेंट्स प्राइवेट लिमिटेड", sanscript.DEVANAGARI, sanscript.ITRANS)
t1 = time.time()
print(f"\nSpeed test: {n_iter} Devanagari transliterations took {t1 - t0:.3f}s ({(t1 - t0) / n_iter * 1000:.3f} ms/item, {n_iter / (t1 - t0):.1f} items/sec)")
