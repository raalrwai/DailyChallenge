def lengthOfLongestSubstring(s: str) -> int:
    i = 0
    j = 1
    seen = set()
    maxSub = 0
    seen.add(s[0])
    while i < len(s) and j < len(s):
        if s[j] not in seen:
            seen.add(s[j])
            j +=1
            maxSub = max(maxSub, j-i)
        else:
            seen.remove(s[i])
            i +=1
    return maxSub
                

print(lengthOfLongestSubstring('abcabcbb'))