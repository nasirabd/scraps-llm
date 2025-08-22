import re
#Look for one or more whitespace character (tabs, newline, spaces) in the string object 
_whitespace = re.compile(r"\s+")

#Clean up whitespaces
def clean(s:str) -> str:
    #remove leading and trailing whitespaces
    s = s.strip()
    #replace the whitespace characters in s with a single space
    s = _whitespace.sub(" ", s) 
    return s
