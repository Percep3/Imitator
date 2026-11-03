"""Descriptive alignment diagnostics; never changes predictions or scoring."""
from collections import Counter, defaultdict


def alignment(pred, target):
    """Minimum edit path, deterministic tie-break: diagonal, deletion, insertion."""
    n, m = len(target), len(pred)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i][j] = min(cost[i-1][j-1] + (target[i-1] != pred[j-1]),
                             cost[i-1][j]+1, cost[i][j-1]+1)
    events, i, j = [], n, m
    while i or j:
        if i and j and cost[i][j] == cost[i-1][j-1] + (target[i-1] != pred[j-1]):
            if target[i-1] != pred[j-1]:
                events.append(dict(op="substitution", position=i-1, target=target[i-1], pred=pred[j-1]))
            i, j = i-1, j-1
        elif i and cost[i][j] == cost[i-1][j]+1:
            events.append(dict(op="deletion", position=i-1, target=target[i-1], pred=None))
            i -= 1
        else:
            events.append(dict(op="insertion", position=i, target=None, pred=pred[j-1]))
            j -= 1
    return list(reversed(events))


def error_taxonomy(rows):
    counts, failures = Counter(), Counter()
    strata = {name: defaultdict(Counter) for name in ("token", "position", "sign_count", "frame_duration", "fold")}
    for row in rows:
        events = alignment(row["pred"], row["target"])
        for event in events:
            counts[event["op"]] += 1
            keys = dict(token=f'{event["target"]}->{event["pred"]}', position=str(event["position"]),
                        sign_count=str(row["sign_count"]), frame_duration=str(row["frame_length"]),
                        fold=str(row["fold"]))
            for name, key in keys.items():
                strata[name][key][event["op"]] += 1
        kinds = sorted({e["op"] for e in events})
        failures["+".join(kinds) if kinds else ("exact" if row["strict_exact"] else "eos_only")] += 1
        counts["premature_eos_samples"] += row["emitted_eos"] and len(row["pred"]) < len(row["target"])
        counts["missing_eos_samples"] += not row["emitted_eos"]
        counts["pred_adjacent_equal_ids"] += sum(a == b for a,b in zip(row["pred"], row["pred"][1:]))
        counts["target_adjacent_equal_ids"] += sum(a == b for a,b in zip(row["target"], row["target"][1:]))
    return dict(n=len(rows), operations=dict(counts),
                ranked_sequence_classes=[dict(kind=k, count=v, prevalence=v/len(rows)) for k,v in failures.most_common()],
                strata={k:{s:dict(c) for s,c in v.items()} for k,v in strata.items()},
                cautions=["Adjacent equal IDs do not establish pathological copying or repeated signs.",
                          "Early EOS is operationally EOS with fewer IDs than reference; it is not a causal attribution.",
                          "Duration is measured in frames; no FPS assumption.",
                          "Alignment tie-break affects operation attribution, never Levenshtein distance."])
