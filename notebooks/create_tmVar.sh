#/bin/bash

for fn in "../cache/rsid/*.txt"; do
   name=`echo $fn | cut -f 1 -d '.'`
   echo $name
done


