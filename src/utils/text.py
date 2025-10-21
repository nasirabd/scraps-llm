import re
#Look for one or more whitespace character (tabs, newline, spaces) in the string object 
_whitespace = re.compile(r"\s+")
_url = re.compile(r"https?://\S+")
_bullets = re.compile(r"^[•\-\*]\s*", flags=re.MULTILINE)

#Clean up whitespaces
def clean(s:str) -> str:
    #remove leading and trailing whitespaces
    s = s.strip()
    # remove URLs
    s = _url.sub("", s)   
     # drop leading bullets             
    s = _bullets.sub("", s)            
    #replace the whitespace characters in s with a single space
    s = _whitespace.sub(" ", s) 
    return s
