from neural_morph_segm import load_cls

import sys


def getRoots(words):
    ans = []
    for word, (labels, probs) in zip(words, model._predict_probs(words)):
        morphemes, morpheme_probs, morpheme_types = model.labels_to_morphemes(
            word, labels, probs, return_probs=True, return_types=True)

        ans.append([(morphem, prob) for morphem, prob, morhem_type in zip(
            morphemes, morpheme_probs, morpheme_types) if morhem_type == 'ROOT'])

    return ans


def getBestRoot(res):
    return max(res, key=lambda x: x[1])


def getSubstring(str1, str2):
    from difflib import SequenceMatcher
    match = SequenceMatcher(None, str1, str2).find_longest_match(
        0, len(str1), 0, len(str2))
    return str1[match.a: match.a + match.size]


def getInput(word1, word2):
    word1, word2 = word1.lower(), word2.lower()
    ans = getRoots([word1, word2])
    roots = [getBestRoot(a)[0] for a in ans]
    substr = getSubstring(*roots)
    if not substr:
        return {
            "prefix_1": roots[0],
            "postfix_1": "",
            "prefix_2": roots[1],
            "postfix_2": "",
            "substr_len": len(substr)
        }
    roots_splited = [r.split(substr) for r in roots]
    return {
        "prefix_1": roots_splited[0][0],
        "postfix_1": roots_splited[0][1],
        "prefix_2": roots_splited[1][0],
        "postfix_2": roots_splited[1][1],
        "substr_len": len(substr)
    }


if __name__ == "__main__":
    model = load_cls("models/morphemes-3-5-3-memo.json")
    words = input("Введите два слова: ").split()

    words_list = list(map(str.strip, words))

    print(getInput(*words_list))
