#!/bin/bash

cat /dev/stdin | \
sed 's: \([A-Z]\) \([A-Z]\) : \1_\2 :g;
     s: \([A-Z]\) \([A-Z]\)$: \1_\2:g;
     s:^\([A-Z]\) \([A-Z]\) :\1_\2 :g;
     s:^\([A-Z]\) \([A-Z]\)$:\1_\2:g;
     s:_\([A-Z]\) \([A-Z]\) :_\1_\2 :g;
     s:_\([A-Z]\) \([A-Z]\)$:_\1_\2:g;
     s:_\([A-Z]\) \([A-Z]\)_:_\1_\2_:g;
     s: \([A-Z]\) \([A-Z]\)_: \1_\2_:g;
     ' | \
sed 's: \([A-Z]\) \([A-Z]\) : \1_\2 :g;
     s: \([A-Z]\) \([A-Z]\)$: \1_\2:g;
     s:^\([A-Z]\) \([A-Z]\) :\1_\2 :g;
     s:^\([A-Z]\) \([A-Z]\)$:\1_\2:g;
     s:_\([A-Z]\) \([A-Z]\) :_\1_\2 :g;
     s:_\([A-Z]\) \([A-Z]\)$:_\1_\2:g;
     s:_\([A-Z]\) \([A-Z]\)_:_\1_\2_:g;
     s: \([A-Z]\) \([A-Z]\)_: \1_\2_:g;
     ' >/dev/stdout

