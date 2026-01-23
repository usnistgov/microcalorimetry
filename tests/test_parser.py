import microcalorimetry.measurements.rfsweep._parser as parser
from pathlib import Path

LOCAL = Path(__file__).parents[0]


def test_merge_dicts():
    d1 = {
        'a': 1,
        'b': 2,
        'c': 3,
        'd': {'fancy': 'cat'},
        'e': {'fancier': {'fancy cat': 'd1'}},
    }

    d2 = {
        'c': 17,
        'd': 2,
        'e': {'fancier': {'fancy cat': 'd2'}, 'secret': {'secret cat': 'not a cat'}},
    }

    d3 = {'d': 'dog'}

    output_a = {
        'a': 1,
        'b': 2,
        'd': {'fancy': 'cat'},
        'c': 3,
        'e': {'secret': {'secret cat': 'not a cat'}, 'fancier': {'fancy cat': 'd1'}},
    }

    output_b = {
        'a': 1,
        'b': 2,
        'd': 2,
        'c': 17,
        'e': {'secret': {'secret cat': 'not a cat'}, 'fancier': {'fancy cat': 'd2'}},
    }

    output_c = {
        'a': 1,
        'b': 2,
        'd': 'dog',
        'c': 17,
        'e': {'secret': {'secret cat': 'not a cat'}, 'fancier': {'fancy cat': 'd2'}},
    }

    test_a = parser._merge_dicts([d2, d1])
    test_b = parser._merge_dicts([d1, d2])
    test_c = parser._merge_dicts([d1, d2, d3])
    test_d = parser._merge_dicts([d1])
    test_e = parser._merge_dicts([])

    assert test_a == output_a
    assert test_b == output_b
    assert test_c == output_c
    assert test_d == d1
    assert test_e == {}


if __name__ == '__main__':
    test_merge_dicts()
