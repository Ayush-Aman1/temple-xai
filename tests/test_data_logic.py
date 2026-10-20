import pandas as pd

from templexai.data.commons_images import licence_ok
from templexai.data.make_splits import split_temples
from templexai.data.style_map import StyleMap, normalise
from templexai.device import pick_device
from templexai.schemas import SplitRow


def test_licence_filter():
    assert licence_ok("CC BY-SA 4.0")
    assert licence_ok("CC BY 3.0")
    assert licence_ok("Public domain")
    assert licence_ok("CC0")
    assert licence_ok("CC BY-SA 3.0 and GFDL")
    assert not licence_ok("CC BY-NC 2.0")
    assert not licence_ok("CC BY-ND 2.0")
    assert not licence_ok("Fair use")


def test_split_is_temple_disjoint_and_stratified():
    sizes = {"Nagara": 40, "Dravida": 90, "Deccan_Vesara": 60}
    df = pd.DataFrame([{"temple_id": f"{t[:4]}{i}", "tradition": t} for t, n in sizes.items() for i in range(n)])
    sp = split_temples(df)
    assert sp.temple_id.is_unique
    assert set(sp.split) == {"train", "val", "test"}
    assert (split_temples(df).split == sp.split).all()  # deterministic
    for t, g in sp.groupby("tradition"):
        assert set(g.split) == {"train", "val", "test"}
    SplitRow(**sp.iloc[0].to_dict())


def test_style_map(tmp_path):
    sm = StyleMap(require_verified=False)
    assert sm.classify("Dravidian architecture") == ("Dravida", "dravidian")
    assert sm.classify("[[Hoysala architecture|Hoysala]]")[0] == "Deccan_Vesara"
    assert sm.classify("Karnata Dravida")[0] == "Deccan_Vesara"  # must not fall into Dravida
    assert sm.classify("Hindu temple architecture") == (None, None)
    assert sm.classify("Kalinga Architecture")[0] == "Nagara"
    assert sm.classify("Pancharatna")[0] == "EXCLUDE"
    assert normalise("[[A|b]]") == "b"


def test_device():
    assert pick_device("cpu") == "cpu"
    assert pick_device("auto") in {"cuda", "mps", "cpu"}
