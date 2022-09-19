

def dict_sort(content, K, reverse=True):
    results = {}
    sort_info = sorted(content.items(), key=lambda item: item[1], reverse=reverse)
    for idx, info in enumerate(sort_info):
        if idx < K:
            results[info[0]] = info[1]
        else:
            break

    return results